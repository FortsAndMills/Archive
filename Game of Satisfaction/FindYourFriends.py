import numpy as np
from tqdm import tqdm

# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---------------------------------------------
def friends(A):
    '''по матрице мира считаем, сколько друзей у каждой точки!'''
    same_type = np.ones_like(A)*10
    same_type[1:-1, 1:-1] = ((A[2:, 1:-1] == A[1:-1, 1:-1]).astype(int) +
                            (A[0:-2, 1:-1] == A[1:-1, 1:-1]) +
                            (A[1:-1, 2:] == A[1:-1, 1:-1]) +
                            (A[1:-1, 0:-2] == A[1:-1, 1:-1]) +
                            (A[2:, 2:] == A[1:-1, 1:-1]) +
                            (A[0:-2, 0:-2] == A[1:-1, 1:-1]) +
                            (A[0:-2, 2:] == A[1:-1, 1:-1]) +
                            (A[2:, 0:-2] == A[1:-1, 1:-1]))
    return same_type
    
def friends_of_color(A, color):
    '''по матрице мира считаем, сколько соседей заданного цвета у каждой точки!'''
    of_type = np.zeros_like(A)
    of_type[1:-1, 1:-1] = ((A[2:, 1:-1] == color).astype(int) +
                            (A[0:-2, 1:-1] == color) +
                            (A[1:-1, 2:] == color) +
                            (A[1:-1, 0:-2] == color) +
                            (A[2:, 2:] == color) +
                            (A[0:-2, 0:-2] == color) +
                            (A[0:-2, 2:] == color) +
                            (A[2:, 0:-2] == color))
    return of_type

def neighbours(x, y):
    '''выдаём массив соседних точек'''
    return [(x + s[0], y + s[1]) for s in [np.array([-1, -1]),
                                           np.array([-1, 0]),
                                           np.array([-1, 1]),
                                           np.array([0, -1]),
                                           np.array([0, 1]),
                                           np.array([1, -1]),
                                           np.array([1, 0]),
                                           np.array([1, 1]),]]

# ГРАФИКА -------------------------------------------------------------
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
from JSAnimation.IPython_display import display_animation
from matplotlib import animation

def show_frames(name, frames, satisf):
    """
    generate animation inline notebook:
    frames - list of pictures
    """      
    
    plt.figure(figsize=(frames[0].shape[1] / 9.0, frames[0].shape[0] / 18.0), dpi = 72)
    plt.suptitle(name)
    
    plt.subplot(121)
    patch1 = plt.imshow(frames[0])
    plt.axis('off')
    plt.title("WORLD")
    
    plt.subplot(122)
    patch2 = plt.imshow(satisf[0])
    plt.axis('off')
    plt.title("SATISFACTION MAP")
    
    def animate(i):
        patch1.set_data(frames[i])
        patch2.set_data(satisf[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    display(display_animation(anim, default_mode='once'))

# МОДЕЛЬ МИРА! --------------------------------------------------------
class World():
    def __init__(self, SIZE = 100, COLORS = 2, SATISFACTION_THRESHOLD = 4):
        self.SIZE = SIZE
        self.COLORS = COLORS
        self.SATISFACTION_THRESHOLD = SATISFACTION_THRESHOLD
        
        self.A = np.random.randint(0, COLORS, size=(SIZE + 2, SIZE + 2))
        self.A[0, :] = -1
        self.A[:, 0] = -1
        self.A[-1, :] = -1
        self.A[:, -1] = -1
        
        # сохраняем мир для отображения
        self.world_map = []
        self.dissatisfied_map = []
    
    def where_to_run(self, x, y):
        raise NotImplemented()
    
    def iteration(self):
        # строим матрицу счастья
        self.friends = friends(self.A)
        self.dissatisfied = np.logical_and(self.friends < self.SATISFACTION_THRESHOLD, self.A >= 0)
        
        # заполняем лог
        self.world_map.append(self.A.copy())
        self.dissatisfied_map.append(self.dissatisfied.copy())
        
        # шафлим точки, которые ХОТЯТ ДВИГАТЬСЯ!
        want_to_move = self.dissatisfied.nonzero()
        want_to_move = list(zip(*want_to_move))
        np.random.shuffle(want_to_move)
        
        # делаем пустые клетки несчастными, таким образом пометив, что они ещё не двигались в этот ход:
        self.dissatisfied = np.logical_or(self.dissatisfied, self.A == -2)
        
        for x, y in want_to_move:
            if self.dissatisfied[x][y]:  # если всё ещё несчастна
                # вызываем функцию, которая скажет нам, куда эта точка может бежать.
                candidates = self.where_to_run(x, y)
                # если есть куда
                if len(candidates) > 0:
                    # берём рандомного соседа
                    cx, cy = candidates[np.random.randint(0, len(candidates))]
                    # меняемся с ним местами
                    self.A[cx][cy], self.A[x][y] = self.A[x][y], self.A[cx][cy]
                    # больше поменявшиеся точки в эту итерацию не трогаем
                    self.dissatisfied[cx][cy], self.dissatisfied[x][y] = False, False
