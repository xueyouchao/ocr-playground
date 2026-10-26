"""The two-slot gate, and the fact that it always releases."""
from __future__ import annotations

import pytest

from ocr_playground.application.capacity import ParseSlots, SlotsBusy


def test_two_holders_can_hold_at_once() -> None:
    slots = ParseSlots(2, wait_seconds=0.1)
    with slots.hold(), slots.hold():
        pass


def test_a_saturated_server_refuses_after_the_wait() -> None:
    slots = ParseSlots(1, wait_seconds=0.01)
    with slots.hold(), pytest.raises(SlotsBusy), slots.hold():
        pass


def test_a_failed_request_still_releases_its_slot() -> None:
    slots = ParseSlots(1, wait_seconds=0.01)
    with pytest.raises(RuntimeError), slots.hold():
        raise RuntimeError("the read blew up")

    with slots.hold():
        pass


def test_zero_slots_is_a_programming_error() -> None:
    with pytest.raises(ValueError, match="at least one parse slot"):
        ParseSlots(0)
