import unittest

from shop.delivery import DeliveryService, SettingsLoader


class DeliveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.delivery = DeliveryService(settings_loader=SettingsLoader())

    def test_small_order_has_delivery_fee(self) -> None:
        self.assertEqual(self.delivery.delivery_cost(1500), 300)

    def test_order_above_free_delivery_threshold(self) -> None:
        self.assertEqual(self.delivery.delivery_cost(2500), 0)

    def test_large_order_has_free_delivery(self) -> None:
        self.assertEqual(self.delivery.delivery_cost(4000), 0)
