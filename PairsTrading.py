from math import log, sqrt, exp, pi
from datamodel import Order, OrderDepth, TradingState
from typing import List, Dict, Any, Tuple
import json

# Product Constants and Parameters
class Product:
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"

# Trading parameters for fine-tuning strategies
PARAMS = {
    Product.PICNIC_BASKET1: {
        "mean": 47.82,
        "std": 8.28,
        "z_entry": 1.6,
        "z_exit": 0.4,
        "max_position": 5,
        "spread_std_window": 45,
    },
    Product.PICNIC_BASKET2: {
        "mean": 29.54,
        "std": 4.92,
        "z_entry": 1.6,
        "z_exit": 0.4,
        "max_position": 5,
        "spread_std_window": 45,
    },
    "BASKET_SPREAD": {
        "mean": 168.28,
        "std": 12.50,
        "z_entry": 1.6,
        "z_exit": 0.4,
        "max_position": 4,
        "spread_std_window": 45,
    },
}

# Basket composition weights
BASKET_WEIGHTS = {
    Product.PICNIC_BASKET1: {
        Product.CROISSANTS: 6,
        Product.JAMS: 3,
        Product.DJEMBES: 1
    },
    Product.PICNIC_BASKET2: {
        Product.CROISSANTS: 4,
        Product.JAMS: 2
    }
}

class Trader:
    def __init__(self):
        # Position limits per product
        self.limits = {
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.DJEMBES: 60,
            Product.PICNIC_BASKET1: 60,
            Product.PICNIC_BASKET2: 100,
        }
        
        # Track spread history for basket strategies
        self.spread_history = {
            Product.PICNIC_BASKET1: [],
            Product.PICNIC_BASKET2: [],
            "BASKET_SPREAD": [],
        }

    # Helper Methods
    def get_position(self, state: TradingState, product: str) -> int:
        return state.position.get(product, 0)
    
    def get_weighted_mid_price(self, order_depth: OrderDepth) -> float:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return 0
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol)
    
    def zscore_to_size(self, z: float, max_size: int) -> int:
        capped_z = max(min(abs(z), 5), 1)
        return max(1, int(round((capped_z / 5) * max_size)))

    # Check if a trade would exceed position limits
    def can_execute_trade(self, state: TradingState, product: str, quantity: int) -> bool:
        current_position = self.get_position(state, product)
        new_position = current_position + quantity
        return abs(new_position) <= self.limits[product]

    def can_execute_component_trades(self, state: TradingState, basket_product: str, basket_quantity: int) -> bool:
        weights = BASKET_WEIGHTS[basket_product]
        for component, qty in weights.items():
            component_quantity = basket_quantity * qty
            if not self.can_execute_trade(state, component, component_quantity):
                return False
        return True

    # Basket Trading Methods
    def get_synthetic_basket_depth(
        self, 
        basket_product: str, 
        order_depths: Dict[str, OrderDepth]
    ) -> OrderDepth:
        weights = BASKET_WEIGHTS[basket_product]
        components = list(weights.keys())
        
        synthetic_depth = OrderDepth()
        
        if not all(c in order_depths for c in components):
            return synthetic_depth
            
        try:
            basket_bid_price = 0
            basket_bid_volume = float('inf')
            
            if not all(order_depths[component].sell_orders for component in components):
                return synthetic_depth
        
            for component, qty in weights.items():
                if not order_depths[component].sell_orders:
                    return synthetic_depth
                    
                component_best_ask = min(order_depths[component].sell_orders.keys())
                component_best_ask_vol = abs(order_depths[component].sell_orders[component_best_ask])
                
                basket_bid_price += component_best_ask * qty
                basket_bid_volume = min(basket_bid_volume, component_best_ask_vol // qty)
        
            if basket_bid_volume > 0:
                synthetic_depth.buy_orders[basket_bid_price] = basket_bid_volume
        
            basket_ask_price = 0
            basket_ask_volume = float('inf')
            
            if not all(order_depths[component].buy_orders for component in components):
                return synthetic_depth
        
            for component, qty in weights.items():
                if not order_depths[component].buy_orders:
                    return synthetic_depth
                    
                component_best_bid = max(order_depths[component].buy_orders.keys())
                component_best_bid_vol = order_depths[component].buy_orders[component_best_bid]
                
                basket_ask_price += component_best_bid * qty
                basket_ask_volume = min(basket_ask_volume, component_best_bid_vol // qty)
        
            if basket_ask_volume > 0:
                synthetic_depth.sell_orders[basket_ask_price] = -basket_ask_volume
                
        except (KeyError, ValueError) as e:
            return OrderDepth()
        
        return synthetic_depth
    
    def calculate_basket_spread(
        self,
        basket_product: str,
        order_depths: Dict[str, OrderDepth],
        params: Dict
    ) -> Tuple[float, float]:
        try:
            basket_depth = order_depths[basket_product]
            synthetic_depth = self.get_synthetic_basket_depth(basket_product, order_depths)
            
            if (not basket_depth.buy_orders or not basket_depth.sell_orders or 
                not synthetic_depth.buy_orders or not synthetic_depth.sell_orders):
                return 0, 0
            
            basket_mid = self.get_weighted_mid_price(basket_depth)
            synthetic_mid = self.get_weighted_mid_price(synthetic_depth)
            
            if basket_mid == 0 or synthetic_mid == 0:
                return 0, 0
        
            spread = basket_mid - synthetic_mid
            
            self.spread_history[basket_product].append(spread)
            spread_std_window = params.get("spread_std_window", 45)
            
            if len(self.spread_history[basket_product]) > spread_std_window:
                self.spread_history[basket_product].pop(0)
        
            if len(self.spread_history[basket_product]) >= 5:
                values = self.spread_history[basket_product]
                n = len(values)
                if n == 0:
                    rolling_std = 0
                else:
                    mean = sum(values) / n
                    variance = sum((x - mean) ** 2 for x in values) / n
                    rolling_std = sqrt(variance)
                std_to_use = rolling_std if rolling_std > 0 else params["std"]
                z_score = (spread - params["mean"]) / std_to_use if std_to_use > 0 else 0
            else:
                z_score = (spread - params["mean"]) / params["std"] if params["std"] > 0 else 0
        
            return spread, z_score
        
        except (KeyError, ValueError, ZeroDivisionError) as e:
            return 0, 0
    
    def convert_synthetic_basket_orders(
        self,
        basket_product: str,
        synthetic_orders: List[Order],
        order_depths: Dict[str, OrderDepth]
    ) -> Dict[str, List[Order]]:
        component_orders = {}
        for component in BASKET_WEIGHTS[basket_product]:
            component_orders[component] = []
        
        synthetic_basket_depth = self.get_synthetic_basket_depth(basket_product, order_depths)
        best_bid = max(synthetic_basket_depth.buy_orders.keys()) if synthetic_basket_depth.buy_orders else 0
        best_ask = min(synthetic_basket_depth.sell_orders.keys()) if synthetic_basket_depth.sell_orders else float('inf')
        
        for order in synthetic_orders:
            price = order.price
            quantity = order.quantity
            
            try:
                if quantity > 0 and price >= best_ask:
                    component_prices = {}
                    for component in BASKET_WEIGHTS[basket_product]:
                        if not order_depths[component].sell_orders:
                            return {}
                        component_prices[component] = min(order_depths[component].sell_orders.keys())
                
                elif quantity < 0 and price <= best_bid:
                    component_prices = {}
                    for component in BASKET_WEIGHTS[basket_product]:
                        if not order_depths[component].buy_orders:
                            return {}
                        component_prices[component] = max(order_depths[component].buy_orders.keys())
                else:
                    continue
                
                for component, qty in BASKET_WEIGHTS[basket_product].items():
                    component_order = Order(
                        component,
                        component_prices[component],
                        quantity * qty
                    )
                    component_orders[component].append(component_order)
            except (KeyError, ValueError):
                return {}
        
        if all(len(orders) == 0 for orders in component_orders.values()):
            return {}
            
        return component_orders
    
    def execute_basket_orders(
        self,
        target_position: int,
        basket_product: str,
        current_position: int,
        order_depths: Dict[str, OrderDepth],
        state: TradingState
    ) -> Dict[str, List[Order]]:
        if target_position == current_position:
            return None

        target_quantity = target_position - current_position
        # Check basket position limit
        if not self.can_execute_trade(state, basket_product, target_quantity):
            return None

        # Check component position limits
        if not self.can_execute_component_trades(state, basket_product, target_quantity):
            return None

        basket_order_depth = order_depths[basket_product]
        synthetic_order_depth = self.get_synthetic_basket_depth(basket_product, order_depths)

        if target_position > current_position:
            if not basket_order_depth.sell_orders or not synthetic_order_depth.buy_orders:
                return None
                
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])
            
            synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
            synthetic_bid_volume = abs(synthetic_order_depth.buy_orders[synthetic_bid_price])
            
            orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
            execute_volume = min(orderbook_volume, target_quantity)
            
            if execute_volume <= 0:
                return None
                
            basket_orders = [Order(basket_product, basket_ask_price, execute_volume)]
            synthetic_orders = [Order("SYNTHETIC", synthetic_bid_price, -execute_volume)]
            
            component_orders = self.convert_synthetic_basket_orders(
                basket_product, synthetic_orders, order_depths
            )
            component_orders[basket_product] = basket_orders
            return component_orders

        else:
            if not basket_order_depth.buy_orders or not synthetic_order_depth.sell_orders:
                return None
                
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])
            
            synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(synthetic_order_depth.sell_orders[synthetic_ask_price])
            
            orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
            execute_volume = min(orderbook_volume, abs(target_quantity))
            
            if execute_volume <= 0:
                return None
                
            basket_orders = [Order(basket_product, basket_bid_price, -execute_volume)]
            synthetic_orders = [Order("SYNTHETIC", synthetic_ask_price, execute_volume)]
            
            component_orders = self.convert_synthetic_basket_orders(
                basket_product, synthetic_orders, order_depths
            )
            component_orders[basket_product] = basket_orders
            return component_orders
    
    def basket_arbitrage_strategy(
        self,
        basket_product: str,
        z_score: float,
        state: TradingState,
        params: Dict
    ) -> Dict[str, List[Order]]:
        weights = BASKET_WEIGHTS[basket_product]
        components = list(weights.keys())
        
        required_products = components + [basket_product]
        if not all(p in state.order_depths for p in required_products):
            return {}
        
        basket_position = self.get_position(state, basket_product)
        target_position = 0
        
        if z_score >= params["z_entry"]:
            target_position = -params["max_position"]
        elif z_score <= -params["z_entry"]:
            target_position = params["max_position"]
        elif abs(z_score) < params["z_exit"]:
            target_position = 0
        else:
            return {}
        
        if target_position != basket_position:
            return self.execute_basket_orders(
                target_position,
                basket_product,
                basket_position,
                state.order_depths,
                state
            )
        
        return {}
    
    def calculate_basket_to_basket_spread(
        self,
        order_depths: Dict[str, OrderDepth],
        params: Dict
    ) -> Tuple[float, float]:
        try:
            basket1_depth = order_depths[Product.PICNIC_BASKET1]
            basket2_depth = order_depths[Product.PICNIC_BASKET2]
            
            if (not basket1_depth.buy_orders or not basket1_depth.sell_orders or 
                not basket2_depth.buy_orders or not basket2_depth.sell_orders):
                return 0, 0
            
            basket1_mid = self.get_weighted_mid_price(basket1_depth)
            basket2_mid = self.get_weighted_mid_price(basket2_depth)
            
            spread = basket1_mid - basket2_mid
            
            self.spread_history["BASKET_SPREAD"].append(spread)
            spread_std_window = params.get("spread_std_window", 45)
            
            if len(self.spread_history["BASKET_SPREAD"]) > spread_std_window:
                self.spread_history["BASKET_SPREAD"].pop(0)
            
            if len(self.spread_history["BASKET_SPREAD"]) >= 5:
                values = self.spread_history["BASKET_SPREAD"]
                n = len(values)
                if n == 0:
                    rolling_std = 0
                else:
                    mean = sum(values) / n
                    variance = sum((x - mean) ** 2 for x in values) / n
                    rolling_std = sqrt(variance)
                std_to_use = rolling_std if rolling_std > 0 else params["std"]
                z_score = (spread - params["mean"]) / std_to_use if std_to_use > 0 else 0
            else:
                z_score = (spread - params["mean"]) / params["std"] if params["std"] > 0 else 0
            
            return spread, z_score
            
        except (KeyError, ValueError, ZeroDivisionError) as e:
            return 0, 0
    
    def basket_spread_strategy(
        self,
        z_score: float,
        state: TradingState,
        params: Dict
    ) -> Dict[str, List[Order]]:
        result = {}
        
        if not (Product.PICNIC_BASKET1 in state.order_depths and 
                Product.PICNIC_BASKET2 in state.order_depths):
            return result
        
        basket1_position = self.get_position(state, Product.PICNIC_BASKET1)
        basket2_position = self.get_position(state, Product.PICNIC_BASKET2)
        
        basket1_mid = self.get_weighted_mid_price(state.order_depths[Product.PICNIC_BASKET1])
        basket2_mid = self.get_weighted_mid_price(state.order_depths[Product.PICNIC_BASKET2])
        
        target_basket1_position = 0
        target_basket2_position = 0
        
        position_size = self.zscore_to_size(z_score, params["max_position"])
            
        if z_score > params["z_entry"]:
            target_basket1_position = -position_size
            target_basket2_position = position_size
        elif z_score < -params["z_entry"]:
            target_basket1_position = position_size
            target_basket2_position = -position_size
        elif abs(z_score) < params["z_exit"]:
            target_basket1_position = 0
            target_basket2_position = 0
        else:
            return result
        
        basket1_order_qty = target_basket1_position - basket1_position
        basket2_order_qty = target_basket2_position - basket2_position

        # Check position limits for both baskets
        if not self.can_execute_trade(state, Product.PICNIC_BASKET1, basket1_order_qty):
            return result
        if not self.can_execute_trade(state, Product.PICNIC_BASKET2, basket2_order_qty):
            return result
        
        if basket1_order_qty != 0:
            basket1_price = int(basket1_mid)
            result[Product.PICNIC_BASKET1] = [Order(Product.PICNIC_BASKET1, basket1_price, basket1_order_qty)]
        
        if basket2_order_qty != 0:
            basket2_price = int(basket2_mid)
            result[Product.PICNIC_BASKET2] = [Order(Product.PICNIC_BASKET2, basket2_price, basket2_order_qty)]
        
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
        
        if "spread_history" not in trader_data:
            trader_data["spread_history"] = {
                Product.PICNIC_BASKET1: [],
                Product.PICNIC_BASKET2: [],
                "BASKET_SPREAD": [],
            }
        
        if Product.PICNIC_BASKET1 in state.order_depths:
            if not trader_data["spread_history"].get(Product.PICNIC_BASKET1):
                trader_data["spread_history"][Product.PICNIC_BASKET1] = self.spread_history[Product.PICNIC_BASKET1]
            
            spread, z_score = self.calculate_basket_spread(
                Product.PICNIC_BASKET1,
                state.order_depths,
                PARAMS[Product.PICNIC_BASKET1]
            )
            
            self.spread_history[Product.PICNIC_BASKET1] = trader_data["spread_history"][Product.PICNIC_BASKET1]
            
            basket1_orders = self.basket_arbitrage_strategy(
                Product.PICNIC_BASKET1,
                z_score,
                state,
                PARAMS[Product.PICNIC_BASKET1]
            )
            
            if basket1_orders:
                for product, orders in basket1_orders.items():
                    if product not in result:
                        result[product] = orders
                    else:
                        result[product].extend(orders)
        
        if Product.PICNIC_BASKET2 in state.order_depths:
            if not trader_data["spread_history"].get(Product.PICNIC_BASKET2):
                trader_data["spread_history"][Product.PICNIC_BASKET2] = self.spread_history[Product.PICNIC_BASKET2]
            
            spread, z_score = self.calculate_basket_spread(
                Product.PICNIC_BASKET2,
                state.order_depths,
                PARAMS[Product.PICNIC_BASKET2]
            )
            
            self.spread_history[Product.PICNIC_BASKET2] = trader_data["spread_history"][Product.PICNIC_BASKET2]
            
            basket2_orders = self.basket_arbitrage_strategy(
                Product.PICNIC_BASKET2,
                z_score,
                state,
                PARAMS[Product.PICNIC_BASKET2]
            )
            
            if basket2_orders:
                for product, orders in basket2_orders.items():
                    if product not in result:
                        result[product] = orders
                    else:
                        result[product].extend(orders)
        
        if (Product.PICNIC_BASKET1 in state.order_depths and
            Product.PICNIC_BASKET2 in state.order_depths):
            if not trader_data["spread_history"].get("BASKET_SPREAD"):
                trader_data["spread_history"]["BASKET_SPREAD"] = self.spread_history["BASKET_SPREAD"]
            
            spread, z_score = self.calculate_basket_to_basket_spread(
                state.order_depths,
                PARAMS["BASKET_SPREAD"]
            )
            
            self.spread_history["BASKET_SPREAD"] = trader_data["spread_history"]["BASKET_SPREAD"]
            
            basket_spread_orders = self.basket_spread_strategy(
                z_score,
                state,
                PARAMS["BASKET_SPREAD"]
            )
            
            for product, orders in basket_spread_orders.items():
                if product not in result:
                    result[product] = orders
                else:
                    order_by_price = {}
                    for order in result[product]:
                        if order.price not in order_by_price:
                            order_by_price[order.price] = order.quantity
                        else:
                            order_by_price[order.price] += order.quantity
                    
                    for order in orders:
                        if order.price not in order_by_price:
                            order_by_price[order.price] = order.quantity
                        else:
                            order_by_price[order.price] += order.quantity
                    
                    result[product] = [
                        Order(product, price, qty) 
                        for price, qty in order_by_price.items()
                        if qty != 0
                    ]
        
        trader_data["spread_history"] = self.spread_history
        traderData = json.dumps(trader_data)
        
        return result, conversions, traderData