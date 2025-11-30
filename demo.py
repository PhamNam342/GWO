import numpy as np
import matplotlib.pyplot as plt
import random

class ImprovedGWO_VRP:
    def __init__(self, num_wolves=40, max_iter=200, num_customers=15):
        self.num_wolves = num_wolves
        self.max_iter = max_iter
        self.num_customers = num_customers
        
        self.depot = np.array([250, 250])
        self.customers = self._initialize_customers()
        self.wolves = []
        self.alpha = None
        self.beta = None
        self.delta = None
        self.convergence_curve = []
        
    def _initialize_customers(self):
        customers = []
        for i in range(self.num_customers):
            customers.append({
                'id': i,
                'x': random.uniform(50, 450),
                'y': random.uniform(50, 450),
                'demand': random.randint(5, 25)
            })
        return customers
    
    def distance(self, p1, p2):
        """Tính khoảng cách Euclidean"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def calculate_total_distance(self, route):
        """Tính tổng quãng đường của lộ trình"""
        if len(route) == 0:
            return float('inf')
        
        total = 0
        customer_0 = np.array([self.customers[route[0]]['x'], self.customers[route[0]]['y']])
        total += self.distance(self.depot, customer_0)
        for i in range(len(route) - 1):
            c1 = np.array([self.customers[route[i]]['x'], self.customers[route[i]]['y']])
            c2 = np.array([self.customers[route[i+1]]['x'], self.customers[route[i+1]]['y']])
            total += self.distance(c1, c2)
        
        customer_last = np.array([self.customers[route[-1]]['x'], self.customers[route[-1]]['y']])
        total += self.distance(customer_last, self.depot)
        
        return total
    
    def initialize_wolves(self):
        """Khởi tạo quần thể sói"""
        for i in range(self.num_wolves):
            route = list(range(self.num_customers))
            random.shuffle(route)
            
            fitness = self.calculate_total_distance(route)
            self.wolves.append({'route': route, 'fitness': fitness})
        self.wolves.sort(key=lambda x: x['fitness'])
        self.alpha = self.wolves[0].copy()
        self.beta = self.wolves[1].copy()
        self.delta = self.wolves[2].copy()
        self.convergence_curve.append(self.alpha['fitness'])
    
    def order_crossover(self, parent1, parent2):
        """Order Crossover (OX)"""
        size = len(parent1)
        start = random.randint(0, size - 1)
        end = random.randint(start, size - 1)
        
        child = [-1] * size
        for i in range(start, end + 1):
            child[i] = parent1[i]
        
        current_pos = (end + 1) % size
        for i in range(size):
            parent2_pos = (end + 1 + i) % size
            if parent2[parent2_pos] not in child:
                child[current_pos] = parent2[parent2_pos]
                current_pos = (current_pos + 1) % size
        
        return child
    
    def swap_mutation(self, route):
        """Swap Mutation"""
        new_route = route.copy()
        i, j = random.sample(range(len(route)), 2)
        new_route[i], new_route[j] = new_route[j], new_route[i]
        return new_route
    
    def two_opt(self, route):
        """2-opt Local Search"""
        improved = True
        best_route = route.copy()
        best_distance = self.calculate_total_distance(best_route)
        
        iterations = 0
        max_iterations = 50 
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            for i in range(1, len(route) - 1):
                for j in range(i + 1, len(route)):
                    new_route = best_route.copy()
                    new_route[i:j+1] = reversed(new_route[i:j+1])
                    
                    new_distance = self.calculate_total_distance(new_route)
                    if new_distance < best_distance:
                        best_route = new_route
                        best_distance = new_distance
                        improved = True
        
        return best_route
    
    def update_wolves(self, iteration):
        """Cập nhật vị trí các sói"""
        a = 2 - iteration * (2 / self.max_iter)
        
        new_wolves = []
        for wolf in self.wolves:
            r = random.random()
            
            if r < 0.5:
                new_route = self.order_crossover(wolf['route'], self.alpha['route'])
            elif r < 0.75:
                new_route = self.order_crossover(wolf['route'], self.beta['route'])
            else:
                new_route = self.order_crossover(wolf['route'], self.delta['route'])
            
            # Mutation với xác suất phụ thuộc vào a
            if random.random() < a / 2:
                new_route = self.swap_mutation(new_route)
                
                if random.random() < 0.2:
                    new_route = self.swap_mutation(new_route)
            if random.random() < 0.3 * (1 - iteration / self.max_iter):
                new_route = self.two_opt(new_route)
            
            new_fitness = self.calculate_total_distance(new_route)
            
            if new_fitness < wolf['fitness']:
                new_wolves.append({'route': new_route, 'fitness': new_fitness})
            else:
                new_wolves.append(wolf)
        
        self.wolves = new_wolves
        
        self.wolves.sort(key=lambda x: x['fitness'])
        self.alpha = self.wolves[0].copy()
        self.beta = self.wolves[1].copy()
        self.delta = self.wolves[2].copy()
        self.convergence_curve.append(self.alpha['fitness'])
    
    def optimize(self):
        """Chạy thuật toán tối ưu"""
        print("=" * 60)
        print("IMPROVED GREY WOLF OPTIMIZER - VEHICLE ROUTING PROBLEM")
        print("=" * 60)
        print(f"Số lượng sói: {self.num_wolves}")
        print(f"Số lần lặp tối đa: {self.max_iter}")
        print(f"Số khách hàng: {self.num_customers}")
        print("=" * 60)
        print()
        
        self.initialize_wolves()
        
        print(f"Lặp 0: Quãng đường tốt nhất = {self.alpha['fitness']:.2f} km")
        
        for iteration in range(1, self.max_iter + 1):
            self.update_wolves(iteration)
            
            if iteration % 20 == 0 or iteration == self.max_iter:
                print(f"Lặp {iteration}: Quãng đường tốt nhất = {self.alpha['fitness']:.2f} km")
        
        print()
        print("=" * 60)
        print("KẾT QUẢ CUỐI CÙNG")
        print("=" * 60)
        print(f"Quãng đường Alpha (tốt nhất): {self.alpha['fitness']:.2f} km")
        print(f"Quãng đường Beta: {self.beta['fitness']:.2f} km")
        print(f"Quãng đường Delta: {self.delta['fitness']:.2f} km")
        print(f"\nLộ trình tối ưu: {[i+1 for i in self.alpha['route']]}")
        print("=" * 60)
    
    def plot_results(self):
        """Vẽ đồ thị kết quả"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Vẽ lộ trình tối ưu
        ax1.set_title('Lộ trình tối ưu', fontsize=14, fontweight='bold')
        ax1.set_xlabel('X (km)')
        ax1.set_ylabel('Y (km)')
        ax1.grid(True, alpha=0.3)
        
        ax1.plot(self.depot[0], self.depot[1], 'rs', markersize=15, label='Depot')
        ax1.text(self.depot[0], self.depot[1] - 20, 'DEPOT', ha='center', fontweight='bold')
        
        # Vẽ khách hàng
        for i, customer in enumerate(self.customers):
            ax1.plot(customer['x'], customer['y'], 'bo', markersize=8)
            ax1.text(customer['x'], customer['y'] - 15, str(i+1), ha='center', fontsize=9)
        
        # Vẽ lộ trình
        route = self.alpha['route']
        x_coords = [self.depot[0]]
        y_coords = [self.depot[1]]
        
        for customer_id in route:
            x_coords.append(self.customers[customer_id]['x'])
            y_coords.append(self.customers[customer_id]['y'])
        
        x_coords.append(self.depot[0])
        y_coords.append(self.depot[1])
        
        ax1.plot(x_coords, y_coords, 'g-', linewidth=2, alpha=0.7, label='Lộ trình')
        ax1.legend()
        
        # Vẽ đường hội tụ
        ax2.set_title('Đường hội tụ', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Số lần lặp')
        ax2.set_ylabel('Quãng đường tốt nhất (km)')
        ax2.plot(self.convergence_curve, 'b-', linewidth=2)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('gwo_vrp_result.png', dpi=300, bbox_inches='tight')
        print("\nĐã lưu kết quả vào file: gwo_vrp_result.png")
        plt.show()


if __name__ == "__main__":
    gwo = ImprovedGWO_VRP(num_wolves=50, max_iter=200, num_customers=15)
    gwo.optimize()
    gwo.plot_results()