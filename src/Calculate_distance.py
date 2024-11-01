import math

class Calculate_distance():

    def __init__(self):

        self.distance_methods = {
            "euclidean": self.euclidean_distance,
            "city_block": self.manhattan_distance,
            "manhattan": self.manhattan_distance,
            "chessboard": self.chessboard_distance,
            "chebyshev": self.chessboard_distance
        }


    def calculate_distance(self, distance_type, point1, point2):
        
        method = self.distance_methods.get(distance_type)
        
        if method:

            return method(point1, point2)
        
        else:

            raise ValueError(f"Unknown distance type: '{distance_type}'")
        


    def euclidean_distance(self, point_1, point_2):

        return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point_1, point_2)))
    
    
    def manhattan_distance(self, point_1, point_2):

        return sum(abs(p1 - p2) for p1, p2 in zip(point_1, point_2))
    

    def chessboard_distance(self, point_1, point_2):

        return max(abs(a - b) for a, b in zip(point_1, point_2))
    

