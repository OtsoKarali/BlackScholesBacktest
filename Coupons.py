from math import log, sqrt, exp, pi
from datamodel import Order, OrderDepth, TradingState
from typing import List, Dict, Any, Tuple
import json

# Product Constants and Parameters
class Product:
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"

# Trading parameters for fine-tuning strategies
PARAMS = {
    Product.VOLCANIC_ROCK_VOUCHER_9500: {
        "mean_volatility": 0.154105,
        "strike": 9500,
        "starting_time_to_expiry": 7/250,
        "std_window": 6,
        "zscore_threshold": 1.49,
    },
    Product.VOLCANIC_ROCK_VOUCHER_9750: {
        "mean_volatility": 0.161,
        "strike": 9750,
        "starting_time_to_expiry": 7/250,
        "std_window": 6,
        "zscore_threshold": 3.29,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10000: {
        "mean_volatility": 0.147766,
        "strike": 10000,
        "starting_time_to_expiry": 7/250,
        "std_window": 6,
        "zscore_threshold": 1.85,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10250: {
        "mean_volatility": 0.138391,
        "strike": 10250,
        "starting_time_to_expiry": 7/250,
        "std_window": 6,
        "zscore_threshold": 1.94,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10500: {
        "mean_volatility": 0.141255,
        "strike": 10500,
        "starting_time_to_expiry": 7/250,
        "std_window": 6,
        "zscore_threshold": 2.05,
    },
}

# Black-Scholes Model for Options Pricing
class BlackScholes:
    @staticmethod
    def norm_cdf(x):
        # Abramowitz and Stegun approximation for erf(z)
        def erf(z):
            if z < 0:
                return -erf(-z)
            a1, a2, a3, a4 = 0.0705230784, 0.0422820123, 0.0092705272, 0.0001520143
            t = 1.0 / (1.0 + a1 * z + a2 * z**2 + a3 * z**3 + a4 * z**4)
            return 1.0 - t**4

        # Normal CDF using erf
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    @staticmethod
    def black_scholes_call(spot, strike, time_to_expiry, volatility):
        if time_to_expiry <= 0 or volatility <= 0 or spot <= 0:
            return 0
        d1 = (log(spot/strike) + (0.5 * volatility * volatility) * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)
        call_price = spot * BlackScholes.norm_cdf(d1) - strike * BlackScholes.norm_cdf(d2)
        return call_price

    @staticmethod
    def delta(spot, strike, time_to_expiry, volatility):
        if time_to_expiry <= 0 or volatility <= 0 or spot <= 0:
            return 0
        d1 = (log(spot/strike) + (0.5 * volatility * volatility) * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        return BlackScholes.norm_cdf(d1)

    @staticmethod
    def implied_volatility(call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-10):
        if call_price <= 0 or spot <= 0 or time_to_expiry <= 0:
            return None  # Replace np.nan with None
        low_vol = 0.01
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0
        for _ in range(max_iterations):
            estimated_price = BlackScholes.black_scholes_call(spot, strike, time_to_expiry, volatility)
            if estimated_price is None:
                return None
            diff = estimated_price - call_price
            if abs(diff) < tolerance:
                return volatility
            elif diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility

class Trader:
    def __init__(self):
        # Position limits per product
        self.limits = {
            Product.VOLCANIC_ROCK: 400,
            Product.VOLCANIC_ROCK_VOUCHER_9500: 200,
            Product.VOLCANIC_ROCK_VOUCHER_9750: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10000: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10250: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10500: 200,
        }

    # Helper Methods
    def get_position(self, state: TradingState, product: str) -> int:
        return state.position.get(product, 0)
    
    def get_mid_price(self, order_depth: OrderDepth) -> float:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return 0
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2

    # Volcanic Rock Voucher Methods
    def get_volcanic_voucher_mid_price(
        self,
        voucher_order_depth: OrderDepth,
        trader_data: Dict,
        voucher_product: str
    ) -> float:
        if (len(voucher_order_depth.buy_orders) > 0 and len(voucher_order_depth.sell_orders) > 0):
            best_bid = max(voucher_order_depth.buy_orders.keys())
            best_ask = min(voucher_order_depth.sell_orders.keys())
            trader_data[f"{voucher_product}_prev_price"] = (best_bid + best_ask) / 2
            return (best_bid + best_ask) / 2
        else:
            return trader_data.get(f"{voucher_product}_prev_price", 0)

    def delta_hedge_volcanic_rock(
        self,
        rock_order_depth: OrderDepth,
        total_voucher_position: int,
        rock_position: int,
        rock_buy_orders: int,
        rock_sell_orders: int,
        total_delta: float
    ) -> List[Order]:
        target_rock_position = -int(total_delta * total_voucher_position)
        hedge_quantity = target_rock_position - (rock_position + rock_buy_orders - rock_sell_orders)

        orders: List[Order] = []
        if hedge_quantity > 0:
            best_ask = min(rock_order_depth.sell_orders.keys())
            quantity = min(abs(hedge_quantity), -rock_order_depth.sell_orders[best_ask])
            quantity = min(quantity, self.limits[Product.VOLCANIC_ROCK] - (rock_position + rock_buy_orders))
            if quantity > 0:
                orders.append(Order(Product.VOLCANIC_ROCK, best_ask, quantity))
        elif hedge_quantity < 0:
            best_bid = max(rock_order_depth.buy_orders.keys())
            quantity = min(abs(hedge_quantity), rock_order_depth.buy_orders[best_bid])
            quantity = min(quantity, self.limits[Product.VOLCANIC_ROCK] + (rock_position - rock_sell_orders))
            if quantity > 0:
                orders.append(Order(Product.VOLCANIC_ROCK, best_bid, -quantity))

        return orders

    def volcanic_voucher_orders(
        self,
        voucher_order_depths: Dict[str, OrderDepth],
        voucher_positions: Dict[str, int],
        trader_data: Dict,
        volatilities: Dict[str, float]
    ) -> Dict[str, List[Order]]:
        result = {}
        max_abs_z_score = 0
        selected_voucher = None
        selected_z_score = 0

        # Calculate z-scores for each voucher and find the one with the highest absolute z-score
        for voucher_product in [
            Product.VOLCANIC_ROCK_VOUCHER_9500,
            Product.VOLCANIC_ROCK_VOUCHER_9750,
            Product.VOLCANIC_ROCK_VOUCHER_10000,
            Product.VOLCANIC_ROCK_VOUCHER_10250,
            Product.VOLCANIC_ROCK_VOUCHER_10500
        ]:
            if voucher_product not in trader_data:
                trader_data[voucher_product] = {
                    "prev_price": 0,
                    "past_vol": []
                }

            volatility = volatilities.get(voucher_product, None)
            if volatility is None:
                continue

            trader_data[voucher_product]["past_vol"].append(volatility)
            if len(trader_data[voucher_product]["past_vol"]) < PARAMS[voucher_product]["std_window"]:
                continue
            if len(trader_data[voucher_product]["past_vol"]) > PARAMS[voucher_product]["std_window"]:
                trader_data[voucher_product]["past_vol"].pop(0)

            # Custom standard deviation calculation
            values = trader_data[voucher_product]["past_vol"]
            n = len(values)
            if n == 0:
                vol_std = 0
            else:
                mean = sum(values) / n
                variance = sum((x - mean) ** 2 for x in values) / n
                vol_std = sqrt(variance)
            if vol_std == 0:
                continue

            z_score = (volatility - PARAMS[voucher_product]["mean_volatility"]) / vol_std
            abs_z_score = abs(z_score)

            if abs_z_score > max_abs_z_score:
                max_abs_z_score = abs_z_score
                selected_voucher = voucher_product
                selected_z_score = z_score

        # Trade the voucher with the highest absolute z-score if it exceeds the threshold
        if selected_voucher and max_abs_z_score >= PARAMS[selected_voucher]["zscore_threshold"]:
            position = voucher_positions[selected_voucher]
            order_depth = voucher_order_depths[selected_voucher]

            if selected_z_score > 0:
                # Sell voucher (implied volatility too high)
                target_position = -self.limits[selected_voucher]
                if position != target_position:
                    target_quantity = abs(target_position - position)
                    if len(order_depth.buy_orders) > 0:
                        best_bid = max(order_depth.buy_orders.keys())
                        quantity = min(target_quantity, order_depth.buy_orders[best_bid])
                        result[selected_voucher] = [Order(selected_voucher, best_bid, -quantity)]
            else:
                # Buy voucher (implied volatility too low)
                target_position = self.limits[selected_voucher]
                if position != target_position:
                    target_quantity = abs(target_position - position)
                    if len(order_depth.sell_orders) > 0:
                        best_ask = min(order_depth.sell_orders.keys())
                        quantity = min(target_quantity, abs(order_depth.sell_orders[best_ask]))
                        result[selected_voucher] = [Order(selected_voucher, best_ask, quantity)]

        return result

    def volcanic_voucher_strategy(
        self,
        state: TradingState,
        trader_data: Dict
    ) -> Dict[str, List[Order]]:
        result = {}
        
        if Product.VOLCANIC_ROCK not in state.order_depths:
            return result

        rock_mid_price = self.get_mid_price(state.order_depths[Product.VOLCANIC_ROCK])
        if rock_mid_price == 0:
            return result

        # Calculate implied volatility and delta for each voucher
        voucher_positions = {}
        volatilities = {}
        total_delta = 0
        total_voucher_position = 0

        for voucher_product in [
            Product.VOLCANIC_ROCK_VOUCHER_9500,
            Product.VOLCANIC_ROCK_VOUCHER_9750,
            Product.VOLCANIC_ROCK_VOUCHER_10000,
            Product.VOLCANIC_ROCK_VOUCHER_10250,
            Product.VOLCANIC_ROCK_VOUCHER_10500
        ]:
            if voucher_product not in state.order_depths:
                continue

            position = self.get_position(state, voucher_product)
            voucher_positions[voucher_product] = position
            total_voucher_position += position

            voucher_mid_price = self.get_volcanic_voucher_mid_price(
                state.order_depths[voucher_product],
                trader_data,
                voucher_product
            )
            if voucher_mid_price == 0:
                continue

            tte = PARAMS[voucher_product]["starting_time_to_expiry"] - (state.timestamp / 1_000_000 / 250)
            if tte <= 0:
                tte = 1 / (250 * 100)

            volatility = BlackScholes.implied_volatility(
                voucher_mid_price,
                rock_mid_price,
                PARAMS[voucher_product]["strike"],
                tte
            )
            volatilities[voucher_product] = volatility

            delta = BlackScholes.delta(
                rock_mid_price,
                PARAMS[voucher_product]["strike"],
                tte,
                volatility if volatility is not None else PARAMS[voucher_product]["mean_volatility"]
            )
            total_delta += delta * position

        # Generate voucher orders
        voucher_orders = self.volcanic_voucher_orders(
            state.order_depths,
            voucher_positions,
            trader_data,
            volatilities
        )

        # Delta hedge the total position
        rock_position = self.get_position(state, Product.VOLCANIC_ROCK)
        rock_orders = self.delta_hedge_volcanic_rock(
            state.order_depths[Product.VOLCANIC_ROCK],
            total_voucher_position,
            rock_position,
            0,  # Assuming no other rock orders in this iteration
            0,
            total_delta
        )

        # Combine orders
        for voucher, orders in voucher_orders.items():
            result[voucher] = orders
        if rock_orders:
            result[Product.VOLCANIC_ROCK] = rock_orders

        return result

    # Main Trading Logic
    def run(self, state: TradingState):
        result = {}
        conversions = 0
        
        trader_data = {}
        if state.traderData:
            try:
                trader_data = json.loads(state.traderData)
            except:
                trader_data = {}
        
        # Volcanic Rock Voucher Strategy
        volcanic_orders = self.volcanic_voucher_strategy(state, trader_data)
        for product, orders in volcanic_orders.items():
            if product not in result:
                result[product] = orders
            else:
                result[product].extend(orders)
        
        traderData = json.dumps(trader_data)
        
        return result, conversions, traderData