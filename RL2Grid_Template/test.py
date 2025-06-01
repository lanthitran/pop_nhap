# Create a list of dictionaries
a = b = [{} for _ in range(2)]  # a and b point to the same list


# Print both a and b
print("a:", a)
print("b:", b)



# Modify the list itself through b
b[0]["y"] = 343433443

# Print both a and b
print("a:", a)
print("b:", b)

# Check if a is b (same list object)
print("a is b:", a is b)

# Check if a[0] is b[0] (same dict object inside)
print("a[0] is b[0]:", a[0] is b[0])
