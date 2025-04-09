#!/usr/bin/env python3

from __future__ import annotations

import logging
import argparse
import os
import sys

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generic, Iterable, Literal, Tuple, TypeVar
from typing_extensions import TypeAlias
from math import sqrt

import numpy as np

# Necessary to load the local gguf package
if (
    "NO_LOCAL_GGUF" not in os.environ
    and (Path(__file__).parent.parent.parent.parent / "gguf-py").exists()
):
    sys.path.insert(0, str(Path(__file__).parent.parent))

from gguf.constants import GGMLQuantizationType

from gguf import GGUFReader, ReaderTensor, dequantize


logger = logging.getLogger("gguf-diff")


_T = TypeVar("_T")


def _eq(x: Any, y: Any) -> tuple[Any, ...]:
    return () if x == y else (x, y)


class TensorDiff2:
    shapes: tuple[tuple[int, ...], tuple[int, ...]]
    types: tuple[GGMLQuantizationType, GGMLQuantizationType]
    mean_err: float
    min_err: float
    max_err: float

    def __init__(self, tensor1: ReaderTensor, tensor2: ReaderTensor):
        self.shapes = (tensor1.shape, tensor2.shape)
        self.types = (tensor1.tensor_type, tensor2.tensor_type)
        if self.shapes[0] == self.shapes[1]:
            # TODO: calculate mse, cos
            pass
        else:
            # FIXME: can't calculate error when shapes differ
            pass

    def __bool__(self) -> bool:
        return (
            (self.shapes[0] == self.shapes[1])
            and (self.types[0] == self.types[1])
            and (self.max_err == 0)
        )


ShapeDiff: TypeAlias = Tuple[Tuple[int, ...], Tuple[int, ...]]
TypeDiff: TypeAlias = Tuple[GGMLQuantizationType, GGMLQuantizationType]


class Diff2(Generic[_T]):
    unique: tuple[dict[str, _T], dict[str, _T]]
    common: dict[str, tuple[_T, _T]]
    shared: dict[str, _T]

    def __init__(
        self,
        a: dict[str, _T],
        b: dict[str, _T],
        cmp_eq: Callable[[_T, _T], tuple[Any, ...]] = _eq,
    ):
        self.unique = ({}, {})
        self.common = {}
        self.shared = {}

        for k, v in a.items():
            if k in b:
                u = b[k]
                if len(cmp_eq(v, u)) == 0:
                    self.shared[k] = v
                else:
                    self.common[k] = (v, u)
            else:
                self.unique[0][k] = v

        for k, v in b.items():
            if k not in a:
                self.unique[1][k] = v


class DiffN(Generic[_T]):
    pass


class GGUFDiff:

    # TODO: how to scale to more than 2 models at a time?
    #       how to display multi-way diffs?
    #       Maybe it doesn't need to be multi-way if the main use-case is to compare against a base model.
    readers: tuple[GGUFReader, GGUFReader]
    meta_diff: Diff2[Any]
    tensor_diff: list[TensorDiff2]

    def __init__(self, *readers: GGUFReader) -> None:
        assert len(readers) == 2, "multi-way diffs are not yet supported"
        self.readers = readers
        meta = [
            {n: v.contents() for n, v in reader.fields.items()} for reader in readers
        ]
        self.meta_diff = Diff2(meta[0], meta[1])
        # FIXME
        self.tensor_diff = [
            TensorDiff2(t1, t2)
            for t1, t2 in zip(readers[0].tensors, readers[1].tensors)
        ]


@dataclass
class NormalizedError:
    min: float
    sum: float
    max: float
    n: int

    @property
    def mean(self) -> float:
        return self.sum / self.n


def diff_dicts(
    a: dict[str, Any], b: dict[str, Any]
) -> list[tuple[tuple[str, Any] | None, tuple[str, Any] | None]]:
    diffs: list[tuple[tuple[str, Any] | None, tuple[str, Any] | None]] = []
    for k in a.keys():
        if k in b:
            if a[k] != b[k]:
                diffs.append(((k, a[k]), (k, b[k])))
        else:
            diffs.append(((k, a[k]), None))
    for k in b.keys():
        if k not in a:
            diffs.append((None, (k, b[k])))
    return diffs


# min, sum and max normalized squared errors
# TODO: use less memory
def diff_tensors(tensor1: ReaderTensor, tensor2: ReaderTensor) -> NormalizedError:
    # Assuming both tensors have the same shape
    t1 = dequantize(tensor1.data, tensor1.tensor_type)
    t2 = dequantize(tensor2.data, tensor2.tensor_type)

    sqerr: np.ndarray = np.square(t1 - t2)
    sq_base = float(np.square(t1).sum().item())
    min = float(sqerr.min().item()) / sq_base
    sum = float(sqerr.sum().item()) / sq_base
    max = float(sqerr.max().item()) / sq_base
    return NormalizedError(min=min, sum=sum, max=max, n=t1.size)


def diff_metadata(reader1: GGUFReader, reader2: GGUFReader) -> bool:
    meta1 = {n: v.contents() for n, v in reader1.fields.items()}
    meta2 = {n: v.contents() for n, v in reader2.fields.items()}
    diff = diff_dicts(meta1, meta2)

    if len(diff) == 0:
        logger.info("Metadata is identical")
        return True

    logger.info("Metadata differs")

    max_name_len = max(len(c[0]) for ab in diff for c in ab if c is not None)

    for a, b in diff:
        if a is not None:
            v = a[1]
            if isinstance(v, list):
                v = np.array(v)
            logger.info(f"< {a[0]:<{max_name_len}} {v}")
        if b is not None:
            v = b[1]
            if isinstance(v, list):
                v = np.array(v)
            logger.info(f"> {b[0]:<{max_name_len}} {v}")

    return False


def diff_tensor_info(reader1: GGUFReader, reader2: GGUFReader) -> bool:
    tensors1: dict[str, ReaderTensor] = {v.name: v for v in reader1.tensors}
    tensors2: dict[str, ReaderTensor] = {v.name: v for v in reader2.tensors}

    identical = True

    max_name_len = max(len(k) for kk in (tensors1.keys(), tensors2.keys()) for k in kk)

    # global min mean and max errors
    g_min_err = None
    g_max_err = None
    g_sum_errs: list[float] = []
    g_n_err: int = 0

    errors: dict[str, NormalizedError] = {}

    for k, t1 in tensors1.items():
        if k in tensors2:
            t2 = tensors2[k]
            if t1.shape.tolist() == t2.shape.tolist():
                err = diff_tensors(t1, t2)
                g_sum_errs.append(err.sum)
                g_n_err += err.n
                g_min_err = err.min if g_min_err is None else min(err.min, g_min_err)
                g_max_err = err.max if g_max_err is None else max(err.max, g_max_err)
                errors[k] = err
                if err.max != 0.0:
                    logger.info(
                        f"  {k:<{max_name_len}} {t1.tensor_type.name} vs {t2.tensor_type.name} err²/norm²: min: {err.min:9.4}, mean: {err.mean:9.4}, max: {err.max:9.4}"
                    )
                    identical = False
            else:
                logger.info(f"< {k:<{max_name_len}} {t1.tensor_type.name} {t1.shape}")
                logger.info(f"> {k:<{max_name_len}} {t2.tensor_type.name} {t2.shape}")
                identical = False

        else:
            logger.info(f"< {k:<{max_name_len}} {t1.tensor_type.name} {t1.shape}")
            identical = False

    for k, t2 in tensors2.items():
        if k not in tensors1:
            logger.info(f"> {k:<{max_name_len}} {t2.tensor_type.name} {t2.shape}")
            identical = False

    g_mean_err = sum(g_sum_errs) / g_n_err if g_n_err > 0 else None

    logger.info("Normalized Errors:")
    logger.info(f"  Min:  {g_min_err}")
    logger.info(f"  Mean: {g_mean_err}")
    logger.info(f"  Max:  {g_max_err}")

    if identical:
        logger.info("Tensors store the exact same values")
    else:
        logger.info("Tensors differ")

    return identical


def diff_models(reader1: GGUFReader, reader2: GGUFReader) -> bool:
    identical: bool = all(
        [
            diff_metadata(reader1, reader2),
            diff_tensor_info(reader1, reader2),
        ]
    )

    return identical


def main():
    parser = argparse.ArgumentParser(
        description="High-level semantic diff of two GGUF files"
    )
    parser.add_argument("model", type=Path, help="GGUF model filename", nargs=2)
    parser.add_argument(
        "--verbose", action="store_true", help="increase output verbosity"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    logger.info(f"* Loading: {args.model[0]}")
    reader1 = GGUFReader(args.model[0], "r")
    logger.info(f"* Loading: {args.model[1]}")
    reader2 = GGUFReader(args.model[1], "r")
    if diff_models(reader1, reader2):
        logger.info("Files are semantically identical")
    else:
        logger.info("Files are semantically different")


if __name__ == "__main__":
    main()
