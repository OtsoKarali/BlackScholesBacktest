from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import List
import jsonpickle
import numpy as np


class Product:
    MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"


PARAMS = {
    Product.MAGNIFICENT_MACARONS: {
        "make_edge": 8.2,
        "make_min_edge": 4.1,
        "make_probability": 0.141,
        "init_make_edge": 8.2,
        "min_edge": 2.1,
        "volume_avg_timestamp": 5,
        "volume_bar": 27,
        "dec_edge_discount": 0.78,
        "step_size": 2.1,
        "position_clear_threshold": 60  # Clear position if it exceeds this
    }
}


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {
            Product.MAGNIFICENT_MACARONS: 75
        }
        self.price_history = []  # Track prices for volatility calculation
        self.traded_volume_history = []  # Track actual traded volume
        self.conversions_used = 0  # Track conversions per timestamp

    def macarons_implied_bid_ask(
        self,
        observation: ConversionObservation,
    ) -> tuple[float, float]:
        return (
            observation.bidPrice - observation.exportTariff - observation.transportFees - 0.8,
            observation.askPrice + observation.importTariff + observation.transportFees
        )

    def macarons_adap_edge(
        self,
        timestamp: int,
        curr_edge: float,
        traded_volume: int,
        traderObject: dict
    ) -> float:
        if timestamp == 0:
            traderObject["MAGNIFICENT_MACARONS"]["curr_edge"] = self.params[Product.MAGNIFICENT_MACARONS]["init_make_edge"]
            return self.params[Product.MAGNIFICENT_MACARONS]["init_make_edge"]

        traderObject["MAGNIFICENT_MACARONS"]["volume_history"].append(traded_volume)
        if len(traderObject["MAGNIFICENT_MACARONS"]["volume_history"]) > self.params[Product.MAGNIFICENT_MACARONS]["volume_avg_timestamp"]:
            traderObject["MAGNIFICENT_MACARONS"]["volume_history"].pop(0)

        if len(traderObject["MAGNIFICENT_MACARONS"]["volume_history"]) < self.params[Product.MAGNIFICENT_MACARONS]["volume_avg_timestamp"]:
            return curr_edge
        elif not traderObject["MAGNIFICENT_MACARONS"]["optimized"]:
            volume_avg = np.mean(traderObject["MAGNIFICENT_MACARONS"]["volume_history"])

            if volume_avg >= self.params[Product.MAGNIFICENT_MACARONS]["volume_bar"]:
                traderObject["MAGNIFICENT_MACARONS"]["volume_history"] = []
                traderObject["MAGNIFICENT_MACARONS"]["curr_edge"] = curr_edge + self.params[Product.MAGNIFICENT_MACARONS]["step_size"]
                return curr_edge + self.params[Product.MAGNIFICENT_MACARONS]["step_size"]

            elif self.params[Product.MAGNIFICENT_MACARONS]["dec_edge_discount"] * self.params[Product.MAGNIFICENT_MACARONS]["volume_bar"] * (curr_edge - self.params[Product.MAGNIFICENT_MACARONS]["step_size"]) > volume_avg * curr_edge:
                if curr_edge - self.params[Product.MAGNIFICENT_MACARONS]["step_size"] > self.params[Product.MAGNIFICENT_MACARONS]["min_edge"]:
                    traderObject["MAGNIFICENT_MACARONS"]["volume_history"] = []
                    traderObject["MAGNIFICENT_MACARONS"]["curr_edge"] = curr_edge - self.params[Product.MAGNIFICENT_MACARONS]["step_size"]
                    traderObject["MAGNIFICENT_MACARONS"]["optimized"] = True
                    return curr_edge - self.params[Product.MAGNIFICENT_MACARONS]["step_size"]
                else:
                    traderObject["MAGNIFICENT_MACARONS"]["curr_edge"] = self.params[Product.MAGNIFICENT_MACARONS]["min_edge"]
                    return self.params[Product.MAGNIFICENT_MACARONS]["min_edge"]

        traderObject["MAGNIFICENT_MACARONS"]["curr_edge"] = curr_edge
        return curr_edge

    def macarons_clear_position(
        self,
        order_depth: OrderDepth,
        position: int,
        orders: List[Order],
        buy_order_volume: int,
        sell_order_volume: int
    ) -> tuple[List[Order], int, int]:
        
        position_limit = self.LIMIT[Product.MAGNIFICENT_MACARONS]
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        if best_bid is None or best_ask is None:
            return orders, buy_order_volume, sell_order_volume

        buy_quantity = position_limit - position
        sell_quantity = position_limit + position

        if position > self.params[Product.MAGNIFICENT_MACARONS]["position_clear_threshold"]:
            clear_quantity = sum(
                volume for price, volume in order_depth.buy_orders.items()
                if price >= best_bid - 2
            )
            clear_quantity = min(clear_quantity, position)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(Product.MAGNIFICENT_MACARONS, best_bid, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        elif position < -self.params[Product.MAGNIFICENT_MACARONS]["position_clear_threshold"]:
            clear_quantity = sum(
                abs(volume) for price, volume in order_depth.sell_orders.items()
                if price <= best_ask + 2
            )
            clear_quantity = min(clear_quantity, abs(position))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(Product.MAGNIFICENT_MACARONS, best_ask, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return orders, buy_order_volume, sell_order_volume

    def macarons_arb_take(
        self,
        order_depth: OrderDepth,
        observation: ConversionObservation,
        adap_edge: float,
        position: int
    ) -> tuple[List[Order], int, int]:
        orders: List[Order] = []
        position_limit = self.LIMIT[Product.MAGNIFICENT_MACARONS]
        buy_order_volume = 0
        sell_order_volume = 0

        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        if best_bid is None or best_ask is None:
            return orders, buy_order_volume, sell_order_volume

        fair_value = (best_bid + best_ask) / 2

        self.price_history.append(fair_value)
        if len(self.price_history) > 20:
            self.price_history.pop(0)

        volatility = np.std(self.price_history) if len(self.price_history) >= 5 else 0

        implied_bid, implied_ask = self.macarons_implied_bid_ask(observation)

        buy_quantity = position_limit - position
        sell_quantity = position_limit + position

        ask = implied_ask + adap_edge

        foreign_mid = (observation.askPrice + observation.bidPrice) / 2
        aggressive_ask = foreign_mid - (2.1 + volatility * 0.5)

        if aggressive_ask > implied_ask:
            ask = aggressive_ask

        edge = (ask - implied_ask) * self.params[Product.MAGNIFICENT_MACARONS]["make_probability"]

        for price in sorted(list(order_depth.sell_orders.keys())):
            if price > implied_bid - edge:
                break

            if price < implied_bid - edge:
                quantity = min(abs(order_depth.sell_orders[price]), buy_quantity)
                if quantity > 0:
                    orders.append(Order(Product.MAGNIFICENT_MACARONS, round(price), quantity))
                    buy_order_volume += quantity

        for price in sorted(list(order_depth.buy_orders.keys()), reverse=True):
            if price < implied_ask + edge:
                break

            if price > implied_ask + edge:
                quantity = min(abs(order_depth.buy_orders[price]), sell_quantity)
                if quantity > 0:
                    orders.append(Order(Product.MAGNIFICENT_MACARONS, round(price), -quantity))
                    sell_order_volume += quantity

        return orders, buy_order_volume, sell_order_volume

    def macarons_arb_clear(
        self,
        position: int
    ) -> int:
        # Respect conversion limit of 10 per timestamp
        conversions = -position
        if abs(conversions) > 10 - self.conversions_used:
            conversions = (10 - self.conversions_used) if conversions > 0 else -(10 - self.conversions_used)
        self.conversions_used += abs(conversions)
        return conversions

    def macarons_arb_make(
        self,
        order_depth: OrderDepth,
        observation: ConversionObservation,
        position: int,
        edge: float,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> tuple[List[Order], int, int]:
        
        orders: List[Order] = []
        position_limit = self.LIMIT[Product.MAGNIFICENT_MACARONS]

        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        if best_bid is None or best_ask is None:
            return orders, buy_order_volume, sell_order_volume

        implied_bid, implied_ask = self.macarons_implied_bid_ask(observation)

        volatility = np.std(self.price_history) if len(self.price_history) >= 5 else 0

        position_skew = (position / position_limit) * 2
        bid = implied_bid - edge - position_skew
        ask = implied_ask + edge + position_skew

        foreign_mid = (observation.askPrice + observation.bidPrice) / 2
        aggressive_ask = foreign_mid - (2.1 + volatility * 0.5)

        if aggressive_ask >= implied_ask + self.params[Product.MAGNIFICENT_MACARONS]['min_edge']:
            ask = aggressive_ask
            print("AGGRESSIVE")
            print(f"ALGO ASK: {round(ask)}")
            print(f"ALGO BID: {round(bid)}")
        else:
            print(f"ALGO ASK: {round(ask)}")
            print(f"ALGO BID: {round(bid)}")

        filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 9]
        filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 18]

        if len(filtered_ask) > 0 and ask > filtered_ask[0]:
            if filtered_ask[0] - 1 > implied_ask:
                ask = filtered_ask[0] - 1
            else:
                ask = implied_ask + edge
        if len(filtered_bid) > 0 and bid < filtered_bid[0]:
            if filtered_bid[0] + 1 < implied_bid:
                bid = filtered_bid[0] + 1
            else:
                bid = implied_bid - edge

        print(f"IMPLIED_BID: {implied_bid}")
        print(f"IMPLIED_ASK: {implied_ask}")
        print(f"FOREIGN ASK: {observation.askPrice}")
        print(f"FOREIGN BID: {observation.bidPrice}")

        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(Product.MAGNIFICENT_MACARONS, round(bid), buy_quantity))

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(Product.MAGNIFICENT_MACARONS, round(ask), -sell_quantity))

        return orders, buy_order_volume, sell_order_volume

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData is not None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}
        self.conversions_used = 0  # Reset conversions used per timestamp

        if Product.MAGNIFICENT_MACARONS in self.params and Product.MAGNIFICENT_MACARONS in state.order_depths:
            if "MAGNIFICENT_MACARONS" not in traderObject:
                traderObject["MAGNIFICENT_MACARONS"] = {"curr_edge": self.params[Product.MAGNIFICENT_MACARONS]["init_make_edge"], "volume_history": [], "optimized": False}
            macarons_position = (
                state.position[Product.MAGNIFICENT_MACARONS]
                if Product.MAGNIFICENT_MACARONS in state.position
                else 0
            )
            print(f"MAGNIFICENT_MACARONS POSITION: {macarons_position}")

            if "last_position" in traderObject:
                traded_volume = abs(macarons_position - traderObject["last_position"])
            else:
                traded_volume = 0
            traderObject["last_position"] = macarons_position
            self.traded_volume_history.append(traded_volume)
            if len(self.traded_volume_history) > 20:
                self.traded_volume_history.pop(0)

            adap_edge = self.macarons_adap_edge(
                state.timestamp,
                traderObject["MAGNIFICENT_MACARONS"]["curr_edge"],
                traded_volume,
                traderObject,
            )

            macarons_take_orders, buy_order_volume, sell_order_volume = self.macarons_arb_take(
                state.order_depths[Product.MAGNIFICENT_MACARONS],
                state.observations.conversionObservations[Product.MAGNIFICENT_MACARONS],
                adap_edge,
                macarons_position,
            )

            macarons_clear_orders, buy_order_volume, sell_order_volume = self.macarons_clear_position(
                state.order_depths[Product.MAGNIFICENT_MACARONS],
                macarons_position,
                macarons_take_orders,
                buy_order_volume,
                sell_order_volume
            )

            macarons_make_orders, _, _ = self.macarons_arb_make(
                state.order_depths[Product.MAGNIFICENT_MACARONS],
                state.observations.conversionObservations[Product.MAGNIFICENT_MACARONS],
                macarons_position,
                adap_edge,
                buy_order_volume,
                sell_order_volume
            )

            # Use conversions to manage position, respecting the limit
            position_after_trades = macarons_position + buy_order_volume - sell_order_volume
            # Avoid conversions if holding a net short position due to storage costs
            if position_after_trades > 0:
                conversions = self.macarons_arb_clear(position_after_trades)
            else:
                conversions = 0

            result[Product.MAGNIFICENT_MACARONS] = (
                macarons_clear_orders + macarons_take_orders + macarons_make_orders
            )

        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData