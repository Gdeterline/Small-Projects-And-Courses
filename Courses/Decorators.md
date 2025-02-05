# **Complete Guide to Python Decorators (with Mini Exercises)** 

Python decorators are a powerful feature that allows you to modify the behavior of functions or methods without changing their code. This guide covers everything from the basics to advanced usage, with exercises to solidify your understanding. 

---

## **1. Understanding Functions as First-Class Objects** 
Before diving into decorators, you must understand that in Python: 
- Functions can be assigned to variables. 
- Functions can be passed as arguments to other functions. 
- Functions can be returned from other functions. 

### **Example: Assigning Functions to Variables** 
```python
def greet():
    return "Hello!"

hello = greet # Assigning function to a variable
print(hello()) # Calling the function via the variable
```

### **Exercise 1: Assign a Function to a Variable** 
Create a function called `square` that returns the square of a number and assign it to another variable before calling it.

---

## **2. Functions Inside Functions (Nested Functions)** 
A function can be defined inside another function. 

### **Example: Nested Functions** 
```python
def outer():
    def inner():
        return "I am inside the outer function."
    return inner # Returning the inner function

fn = outer() # fn now holds the inner function
print(fn()) # Call the returned function
```

### **Exercise 2: Create a Nested Function** 
Write a function `multiplier(n)` that returns another function that multiplies its input by `n`.

---

## **3. Passing Functions as Arguments** 
A function can accept another function as an argument.

### **Example: Function as Argument** 
```python
def apply_function(func, value):
    return func(value)

def double(x):
    return x * 2

print(apply_function(double, 5)) # Output: 10
```

### **Exercise 3: Pass a Function as an Argument** 
Write a function `apply_twice(func, value)` that applies `func` to `value` twice.

---

## **4. First Simple Decorator** 
A **decorator** is just a function that takes another function as input and returns a modified function.

### **Example: Basic Decorator** 
```python
def my_decorator(func):
    def wrapper():
        print("Something before the function runs.")
        func()
        print("Something after the function runs.")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()
```

### **Exercise 4: Create a Simple Decorator** 
Create a decorator `log_decorator` that prints `"Function is running..."` before executing the function.

---

## **5. Using `functools.wraps` to Preserve Metadata** 
When using decorators, function names and docstrings can be lost. Use `functools.wraps` to fix this.

### **Example: Using `wraps`** 
```python
import functools

def my_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print("Before function call")
        result = func(*args, **kwargs)
        print("After function call")
        return result
    return wrapper

@my_decorator
def add(a, b):
    """Returns the sum of two numbers."""
    return a + b

print(add(3, 4)) # Output: 7
print(add.__name__) # Output: add
```

### **Exercise 5: Use `functools.wraps`** 
Modify your `log_decorator` to use `functools.wraps` and preserve function metadata.

---

## **6. Decorators with Arguments** 
You can pass arguments to decorators by wrapping them in another function.

### **Example: Decorator with Arguments** 
```python
def repeat(n):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(n):
                func(*args, **kwargs)
        return wrapper
    return decorator

@repeat(3)
def hello():
    print("Hello!")

hello() # Prints "Hello!" three times
```

### **Exercise 6: Create a Parameterized Decorator** 
Create a decorator `delay(seconds)` that delays the function execution by `seconds` using `time.sleep`.

---

## **7. Applying Multiple Decorators** 
Multiple decorators can be stacked.

### **Example: Stacking Decorators** 
```python
def uppercase(func):
    @functools.wraps(func)
    def wrapper():
        return func().upper()
    return wrapper

def exclaim(func):
    @functools.wraps(func)
    def wrapper():
        return func() + "!"
    return wrapper

@exclaim
@uppercase
def greet():
    return "hello"

print(greet()) # Output: "HELLO!"
```

### **Exercise 7: Stack Decorators** 
Create two decorators `bold` and `italic`, then apply them to a function `greet()` to return a bold and italic string.

---

## **8. Class-Based Decorators** 
A decorator can also be implemented as a class.

### **Example: Class-Based Decorator** 
```python
class CountCalls:
    def __init__(self, func):
        self.func = func
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"Call {self.count} to {self.func.__name__}")
        return self.func(*args, **kwargs)

@CountCalls
def say_hello():
    print("Hello!")

say_hello()
say_hello()
```

**Nota Bene:** It is necessary to name your function `__call__` for the object to be callable.  

### **Exercise 8: Create a Class-Based Decorator** 
Create a class `Timer` that measures and prints the execution time of a function.

---

## **9. Using Built-in Decorators (`@staticmethod`, `@classmethod`, `@property`)** 
Python provides built-in decorators for object-oriented programming.

### **Example: Using `@staticmethod` and `@classmethod`** 
```python
class MathOperations:
    @staticmethod
    def add(a, b):
        return a + b

    @classmethod
    def info(cls):
        return f"This is a {cls.__name__} class."

print(MathOperations.add(3, 5)) # Output: 8
print(MathOperations.info()) # Output: This is a MathOperations class.
```

### **Exercise 9: Create a Class with `@staticmethod` and `@classmethod`** 
Create a class `Car` with a `@staticmethod` to check if a car is fast based on its speed and a `@classmethod` that returns the class name.

---

## **10. Real-World Examples of Decorators** 
Here are some real-world use cases: 

1. **Logging Decorator**: Logs function calls. 
2. **Authentication Decorator**: Checks user authentication before running a function. 
3. **Caching Decorator**: Stores previous results to speed up execution (`functools.lru_cache`). 

### **Example: `functools.lru_cache` for Caching** 
```python
from functools import lru_cache

@lru_cache(maxsize=5)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(10))
```

### **Exercise 10: Use `lru_cache`** 
Modify the Fibonacci function to use `@lru_cache` and test its performance.

---

## **Conclusion** 
Decorators are a fundamental feature of Python that provide flexibility, reusability, and code cleanliness. Understanding them will greatly enhance your ability to write elegant and efficient Python programs.