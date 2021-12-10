from ipywidgets import interact, interactive

def f(x):
    return x

if __name__ == '__main__':
    interact(f, x=(0,20))