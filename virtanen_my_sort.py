x = input('Give a list of integers separated by space: ')

list_x = [int(i) for i in list(x.split(' '))]
sorted_x = sorted(list_x)

print('Given numbers sorted: ' + str(sorted_x))