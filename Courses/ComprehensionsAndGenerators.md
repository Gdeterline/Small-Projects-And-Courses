# **Complete Guide to Comprehensions and Generators in Python** 

Comprehensions and generators are powerful tools that make Python code more concise and efficient. This guide covers list, dictionary, and set comprehensions, as well as generators, with exercises to reinforce learning. 

---

# **1. List Comprehensions** 
List comprehensions provide a concise way to create lists. 

### **Basic Syntax** 
```python
new_list = [expression for item in iterable if condition]
```

### **Example: Creating a List of Squares** 
```python
squares = [x**2 for x in range(10)]
print(squares) # Output: [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

### **Exercise 1: Create a List of Even Numbers** 
Create a list of even numbers from 1 to 20 using list comprehension.

---

## **2. List Comprehensions with Conditions** 
### **Example: Filtering with `if`** 
```python
evens = [x for x in range(20) if x % 2 == 0]
print(evens) # Output: [0, 2, 4, 6, ..., 18]
```

### **Example: If-Else in Comprehensions** 
```python
labels = ["Even" if x % 2 == 0 else "Odd" for x in range(10)]
print(labels) # Output: ['Even', 'Odd', 'Even', 'Odd', ...]
```

### **Exercise 2: Categorize Numbers** 
Write a comprehension that generates `"Positive"` if the number is positive, `"Negative"` if negative, and `"Zero"` otherwise, for numbers from -5 to 5.

---

## **3. Nested List Comprehensions** 
### **Example: Multiplication Table** 
```python
table = [[x * y for x in range(1, 6)] for y in range(1, 6)]
print(table)
```

### **Exercise 3: Flatten a Nested List** 
Given `nested_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]`, use list comprehension to flatten it into a single list.

---

# **4. Dictionary Comprehensions** 
Dictionary comprehensions let you create dictionaries efficiently.

### **Example: Squaring Numbers in a Dictionary** 
```python
squares = {x: x**2 for x in range(5)}
print(squares) # Output: {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
```

### **Exercise 4: Reverse Key-Value Pairs** 
Given `original = {'a': 1, 'b': 2, 'c': 3}`, create a dictionary swapping keys and values.

---

## **5. Dictionary Comprehensions with Conditions** 
### **Example: Filtering in a Dictionary** 
```python
squared_evens = {x: x**2 for x in range(10) if x % 2 == 0}
print(squared_evens) # Output: {0: 0, 2: 4, 4: 16, ...}
```

### **Exercise 5: Filter Students Who Passed** 
Given `grades = {"Alice": 85, "Bob": 60, "Charlie": 72}`, create a dictionary of students who scored at least 70.

---

# **6. Set Comprehensions** 
Set comprehensions create sets concisely.

### **Example: Unique Squares** 
```python
unique_squares = {x**2 for x in range(-3, 4)}
print(unique_squares) # Output: {0, 1, 4, 9}
```

### **Exercise 6: Extract Unique Words** 
Given `sentence = "Python is great and Python is fun"`, create a set of unique words.

---

# **7. Generators** 
Generators are functions that yield values lazily using `yield`. 

### **Example: Generator Function** 
```python
def count_up_to(n):
    count = 1
    while count <= n:
    yield count
    count += 1

counter = count_up_to(5)
print(next(counter)) # Output: 1
print(next(counter)) # Output: 2
```

### **Exercise 7: Fibonacci Generator** 
Write a generator function that produces the Fibonacci sequence.

---

## **8. Generator Expressions** 
Generator expressions provide a memory-efficient way to create generators.

### **Example: Generator Expression for Squares** 
```python
squares = (x**2 for x in range(10))
print(next(squares)) # Output: 0
```

### **Exercise 8: Sum of Squares** 
Use a generator expression to compute the sum of squares of numbers from 1 to 100.

---

# **9. Comparing List Comprehensions and Generators** 
A list comprehension creates a full list in memory, while a generator produces items one at a time.

### **Example: Memory Efficiency** 
```python
import sys

list_comprehension = [x**2 for x in range(1000)]
generator_expression = (x**2 for x in range(1000))

print(sys.getsizeof(list_comprehension)) # Large memory usage
print(sys.getsizeof(generator_expression)) # Small memory usage
```

### **Exercise 9: Compare Memory Usage** 
Use `sys.getsizeof` to compare the memory usage of a list comprehension and a generator expression that compute cubes from 1 to 10,000.

---

# **10. Infinite Generators** 
Infinite generators continue yielding values indefinitely.

### **Example: Infinite Counter** 
```python
def infinite_counter():
    count = 0
    while True:
    yield count
    count += 1

counter = infinite_counter()
print(next(counter)) # Output: 0
print(next(counter)) # Output: 1
```

### **Exercise 10: Infinite Even Numbers** 
Write an infinite generator that yields even numbers starting from 0.

---