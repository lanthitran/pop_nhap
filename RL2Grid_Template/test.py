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

        
