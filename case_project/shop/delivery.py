"""Небольшой расчет доставки для учебного проекта."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DeliverySettings:
    free_delivery_from: int
    delivery_price: int


class SettingsLoader:
    def load(self) -> DeliverySettings:
        try:
            values = json.loads(Path("settings.json").read_text(encoding="utf-8"))
        except FileNotFoundError:
            return DeliverySettings(free_delivery_from=3000, delivery_price=300)
        return DeliverySettings(
            free_delivery_from=values["free_delivery_from"],
            delivery_price=values["delivery_price"],
        )


class DeliveryService:
    def __init__(self, settings_loader: SettingsLoader | None = None) -> None:
        self._settings_loader = (
            settings_loader if settings_loader is not None else SettingsLoader()
        )

    def delivery_cost(self, order_total: int) -> int:
        settings = self._settings_loader.load()
        if order_total >= settings.free_delivery_from:
            return 0
        return settings.delivery_price
