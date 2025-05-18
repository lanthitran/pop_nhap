#laptop, public 
# sss
# private
# private 2
#create a simple class for numerical calculations, pls always define types for variables
class Calculator:
    def add(self, x: float, y: float) -> float:
        """Adds two numbers."""
        return x + y

    def subtract(self, x: float, y: float) -> float:
        """Subtracts two numbers."""
        return x - y

    def multiply(self, x: float, y: float) -> float:
        """Multiplies two numbers."""
        return x * y

    def divide(self, x: float, y: float) -> float:
        """Divides two numbers.  Handles division by zero."""
        if y == 0:
            raise ZeroDivisionError("Cannot divide by zero.")
        return x / y
    def square_root(self, x: float) -> float:
        """Calculates the square root of a number."""
        if x < 0:
            raise ValueError("Cannot calculate the square root of a negative number.")
        return x**0.5

    def log(self, x: float, base: float = 10) -> float:
        """Calculates the logarithm of a number with a specified base."""
        if x <= 0:
            raise ValueError("Cannot calculate the logarithm of a non-positive number.")
        if base <= 0 or base == 1:
            raise ValueError("Invalid base for logarithm.")
        return math.log(x, base)

    def power(self, x: float, y: float) -> float:
        """Calculates x raised to the power of y."""
        return x**y

    def factorial(self, n: int) -> int:
        """Calculates the factorial of a non-negative integer."""
        if not isinstance(n, int) or n < 0:
            raise ValueError("Input must be a non-negative integer.")
        if n == 0:
            return 1
        else:
            return n * self.factorial(n - 1)
import math # is this a bug? fix it

        
# now i want to use the above class, create a simple script to test it
if __name__ == "__main__":
    calc = Calculator()
    print(calc.add(2, 3))
    print(calc.subtract(5, 2))
    print(calc.multiply(4, 6))
    print(calc.divide(10, 2))
    print(calc.divide(10, 0))  # this should raise an error
    #print(calc.divide(10, 0))  # this should raise an error
    #print(calc.divide(10, 0))  # this should raise an error
    #print(calc.divide(10, 0))  # this should raise an error



# 1 
# nhap 1 
# nhap 2 
# nhap 3 
# main 
# nhap 1 again 


# nhap 111111111111111111
# mainnn 


        # nhap 111
# nhap 1 lan 2 
# nhap 1 lan 3



# nhap 