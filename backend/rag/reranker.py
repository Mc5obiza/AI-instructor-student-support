from __future__ import annotations

import logging
import os
from functools import lru_cache

import torch
from sentence_transformers import CrossEncoder


LOGGER = logging.getLogger(__name__)


def resolve_reranker_device(requested_device: str | None = None) -> str:
	"""Resolve the CrossEncoder device from an override or CUDA availability."""
	requested = (requested_device or os.environ.get("RERANKER_DEVICE", "auto")).strip().lower()
	if requested == "auto":
		device = "cuda" if torch.cuda.is_available() else "cpu"
		LOGGER.info("Reranker device auto-selected as %s.", device)
		return device
	if requested == "cpu":
		return "cpu"
	if requested == "cuda":
		if not torch.cuda.is_available():
			raise RuntimeError(
				"RERANKER_DEVICE=cuda was requested, but CUDA is unavailable in this PyTorch runtime."
			)
		return "cuda"
	if requested.startswith("cuda:"):
		if not torch.cuda.is_available():
			raise RuntimeError(
				f"RERANKER_DEVICE={requested} was requested, but CUDA is unavailable "
				"in this PyTorch runtime."
			)
		try:
			device_index = int(requested.partition(":")[2])
		except ValueError as error:
			raise ValueError(f"Invalid reranker device: {requested!r}") from error
		if device_index < 0 or device_index >= torch.cuda.device_count():
			raise ValueError(f"CUDA device index {device_index} is not available.")
		return requested
	raise ValueError(
		"RERANKER_DEVICE must be 'auto', 'cpu', 'cuda', or a specific device such as 'cuda:0'."
	)


@lru_cache(maxsize=1)
def _load_reranker_on_device(
	model_name: str,
	local_files_only: bool,
	device: str,
) -> tuple[CrossEncoder, str]:
	"""Load and retain one CrossEncoder instance for the process lifetime."""
	LOGGER.info("Loading reranker %s on %s.", model_name, device)
	model = CrossEncoder(
		model_name,
		device=device,
		local_files_only=local_files_only,
	)
	return model, device


def load_reranker(
	model_name: str,
	*,
	local_files_only: bool = False,
	requested_device: str | None = None,
) -> tuple[CrossEncoder, str]:
	"""Return the cached CrossEncoder on the resolved device."""
	device = resolve_reranker_device(requested_device)
	return _load_reranker_on_device(model_name, local_files_only, device)


def clear_reranker_cache() -> None:
	"""Release the cached model reference, primarily for controlled tests."""
	_load_reranker_on_device.cache_clear()
