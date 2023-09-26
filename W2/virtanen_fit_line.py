import numpy as np
import matplotlib.pyplot as plt

selected_x = []
selected_y = []

def main():

    select_points()

    a,b = mylinfit(selected_x,selected_y)

    plt.plot(selected_x, selected_y,'ro')
    xp = np.arange(-0,20,0.1)
    plt.plot(xp, a*xp+b,'r-')
    print(f"My_fit : a={b} and b={b}")

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title("fitted line for selected points")
    plt.show()

#linear solver
def mylinfit(x,y):

    # convert list to numpy array
    x = np.array(x)
    y = np.array(y)

    # calculate variables
    N = len(x)
    xy = np.sum(x * y)
    x_sum = np.sum(x)
    y_sum = np.sum(y)
    x_squared_sum = np.sum(x**2)

    a = (N * xy - x_sum * y_sum) / (N * x_squared_sum - x_sum**2)
    b = (y_sum - a * x_sum) / N

    return a, b

def select_points():

    # Create an empty figure and connect the mouse click event handler
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('button_press_event', on_click)

    # Set the axis limits to stay fixed at the specified range
    ax.set_xlim([-1, 20])
    ax.set_ylim([-1, 20])

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Select points with left mouse click (right click to end selection)')

    plt.show()

def on_click(event):
    # Left-click to select a point
    if event.button == 1:  
        selected_x.append(event.xdata)
        selected_y.append(event.ydata)
        plt.plot(event.xdata, event.ydata, 'ro')  # Plot the selected point in red
        plt.draw()

    # Right-click to signal completion
    elif event.button == 3:  
        plt.close()

if __name__ == "__main__":
    main()
