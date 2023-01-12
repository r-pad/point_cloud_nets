import logging
import multiprocessing
import os
import sys
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    runtime_checkable,
)

import numpy as np
import torch
import torch.utils.data as td
import torch_geometric.data as tgd
import tqdm


@runtime_checkable
class CanSample(Protocol):
    """This is just a dataset which has the 'get_data' function. This function doesn't
    need to be deterministic, for instance if we want to randomly sample viewpoints."""

    def get_data(self, *args, seed=None) -> tgd.Data:
        """The"""


class SinglePathDataset(tgd.InMemoryDataset):
    """Dataset consisting of a single file, generated in a different process."""

    def __init__(
        self, processed_path: str, transform=None, pre_transform=None, pre_filter=None
    ):
        root = Path(processed_path).parent
        super().__init__(root, transform, pre_transform, pre_filter, log=False)
        self.data, self.slices = torch.load(processed_path)

    @staticmethod
    def write(data_list: Sequence[tgd.Data], data_path: str):
        """Write a list of data elements into a single file on disk.

        Args:
            data_list (Sequence[tgd.Data]): A list of Data objects.
            data_path (str): The file to write to.
        """
        data, slices = tgd.InMemoryDataset.collate(data_list)
        torch.save((data, slices), data_path)


__dataset: Optional[CanSample] = None


def __can_sample_init(dset_cls, dset_kwargs):
    global __dataset
    __dataset = dset_cls(**dset_kwargs)
    assert isinstance(dset_cls, CanSample)


def __paralel_sample(args: Tuple[int, Tuple, int, str, Any]):
    task_id, get_data_args, n_repeat, processed_dir, seed = args

    if sys.platform == "linux":
        os.sched_setaffinity(os.getpid(), [task_id % os.cpu_count()])  # type: ignore

    assert len(get_data_args) >= 1

    global __dataset
    assert __dataset is not None

    # The output should go in a file that indexes exactly what params
    # were used to generate it.
    prefix = "_".join(get_data_args)
    data_file = f"{prefix}_{n_repeat}.pt"
    data_path = os.path.join(processed_dir, data_file)

    if os.path.exists(data_path):
        logging.info(f"{data_path} already exists, skipping...")
        return True

    try:
        # Sample the data.
        data_list = []
        for _ in range(n_repeat):
            data_list.append(__dataset.get_data(*get_data_args, seed=seed))

        # Save it in an in-memory dataset.
        SinglePathDataset.write(data_list, data_path)

        return True
    except Exception as e:
        logging.error(f"unable to sample get_data({get_data_args}): {e}")
        return False


def parallel_sample(
    dset_cls: Type[CanSample],
    dset_kwargs: Dict[str, Any],
    get_data_args: List[Tuple],
    processed_path: str,
    n_repeat: int,
    n_proc: int = -1,
    seed=None,
):
    """Run parallel sampling in a dataset. Assumes that we're operating on a
    partnet-style dataset, where we have a unique object which may be sampled
    many different times (i.e. different point clouds, different camera positions)

    This works across dataset types; the only restriction on the dataset is that
    it implements CanSample.

    Each object will be sampled n_repeat times, and collated into a single file
    for each object, placed in the processed_dir (defined by the dataset). This
    filename is <OBJ_ID>_<N_REPEAT>.pt

    For downstream datasets, you can load each file into a SingleObjDataset,
    and then use torch.utils.data.ConcatDataset to concatentate them.

    Args:
        dset_cls (Type[CanSample]): The constructor for the dataset.
        dset_kwargs (Dict): The kwargs to pass to the dataset constructor.
        get_data_args (List[Tuple]): The arguments to pass to each call to get_data. If there
            are N distinct objects in the dataset, this parameter should have the form:

            [(OBJ_ID, arg_1, ...), ...]

            where the length of the list is N, and the first entry in each tuple is the
            object ID (a string). These arguments will be passed to get_data via *get_data_args.
        n_repeat (int, optional): Number of times to call get_data on the same args
            (i.e. when get_data is nondeterministic, to sample viewpoints, etc.). Defaults to 100.
        n_proc (int, optional): Number of processes for multiprocessing.
            If 0, multiprocessing won't be used at all (good for debugging serially).
            If -1, uses as many CPUs as are available.
            Defaults to -1.

    Raises:
        ValueError: If any sampling fails, this method will raise an exception.
    """
    if n_proc == -1:
        n_proc = os.cpu_count()  # type: ignore

    n_keys = len(get_data_args)

    # task_number, args, n_repeat, processed_dir, seed
    ps_args = list(
        zip(
            range(n_keys),
            get_data_args,
            [n_repeat] * n_keys,
            [processed_path] * n_keys,
            np.random.SeedSequence(seed).spawn(n_keys),
        )
    )

    # Debugging, means in-memory.
    if n_proc == 0:
        print("sampling with no workers, debug")
        # Initialize the dataset.
        __can_sample_init(dset_cls, dset_kwargs)

        res = [__paralel_sample(args) for args in tqdm.tqdm(ps_args)]

    else:
        multiprocessing.set_start_method("spawn", force=True)
        # Create a multiprocessing pool, and initialize it with a global dataset object
        # that can be used to sample things.
        pool = multiprocessing.Pool(n_proc, __can_sample_init, (dset_cls, dset_kwargs))

        # Perform the parallel mapping.
        res = list(
            tqdm.tqdm(pool.imap(__paralel_sample, ps_args), total=len(get_data_args))
        )

    if not all(res):
        raise ValueError("Sampling failed, please debug.")


class MultiKeyDataset(tgd.Dataset):
    """A hierarchical dataset, where each key has multiple data samples associated with it.

    Consider each key as a unique entity; for instance, the entity might be:
        * A single object
        * A scene, with different settings

    Each entity may be sampled; this sampling may be deterministic, or may be stochastic,
    depending on the setting. For instance, if we're sampling an object which has some internal
    state, sampling multiple times with the same key may yield different results.

    Given a set of entities, we'd like to do normal dataset operation things (i.e. repeat, iterate,
    random access, etc.). However, we'd also like to be able to generate an offline dataset as well,
    for performance reasons. So we'd like to be able to sample on-the-fly, or read from
    memory. In the in-memory configuration, each entity gets its own SinglePathDataset.
    """

    def __init__(
        self,
        dset_cls: Type[CanSample],
        dset_kwargs: Dict[str, Any],
        sample_keys: Sequence[Tuple[str]],
        root: str,
        processed_dirname: str,
        n_repeat: int,
        n_proc: int = -1,
        use_processed: bool = True,
        seed=None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        log: bool = True,
    ):
        for key in sample_keys:
            for arg in key:
                if not isinstance(arg, str):
                    raise ValueError("sample keys have to be strings, for now...")
        # Dataset.
        self._dset_cls = dset_cls
        self._dset_kwargs = dset_kwargs
        self._sample_keys = sample_keys

        # Sampling parameters.
        self._processed_dirname = processed_dirname
        self._n_repeat = n_repeat
        self._n_proc = n_proc
        self._use_processed = use_processed

        # Random seed. Important for reproducibility!
        self._seed = seed

        # This has to come before inmem_dset is created.
        super().__init__(root, transform, pre_transform, pre_filter, log)

        if not self._use_processed:
            # If we don't want to use the processed version (i.e. for on-the-fly loading),
            # we just instantiate.
            self.dataset = dset_cls(**dset_kwargs)
        else:
            self.inmem_dset: td.ConcatDataset = td.ConcatDataset(
                [SinglePathDataset(data_path) for data_path in self.processed_paths]
            )

    @property
    def processed_file_names(self) -> List[str]:
        return [f"{'_'.join(key)}_{self._n_repeat}.pt" for key in self._sample_keys]

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self._processed_dirname)

    def len(self) -> int:
        return len(self._sample_keys) * self._n_repeat

    def get(self, idx: int, seed=None):
        if self._use_processed:
            return self.inmem_dset[idx]
        else:
            sample_key = self._sample_keys[idx // self._n_repeat]
            return self.dataset.get_data(*sample_key, seed=seed)

    def process(self):
        if not self._use_processed:
            return
        else:
            parallel_sample(
                self._dset_cls,
                self._dset_kwargs,
                self._sample_keys,
                self.processed_dir,
                self._n_repeat,
                self._n_proc,
                self._seed,
            )
