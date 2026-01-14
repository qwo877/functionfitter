import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def animate_training(x_dense, y_dense, x_train, y_train, predictions_history, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(x_dense.min(), x_dense.max())
    y_min = min(y_dense.min(), y_train.min()) - 1
    y_max = max(y_dense.max(), y_train.max()) + 1
    ax.set_ylim(y_min, y_max)
    ax.plot(x_dense, y_dense, 'k--', label='True Function', alpha=0.6)
    ax.scatter(x_train, y_train, color='red', s=20, label='Training Points')
    line_pred, = ax.plot([], [], 'b-', lw=3, label='MLP Fitted Line')
    epoch_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    ax.legend(loc='upper right')
    def init():
        line_pred.set_data([], [])
        epoch_text.set_text('')
        return line_pred, epoch_text

    def update(frame):
        current_pred = predictions_history[frame]
        
        sorted_indices = np.argsort(x_train.flatten())
        x_sorted = x_train.flatten()[sorted_indices]
        y_sorted = current_pred.flatten()[sorted_indices]
        
        line_pred.set_data(x_sorted, y_sorted)
        epoch_text.set_text(f'Frame: {frame}')
        return line_pred, epoch_text

    ani = animation.FuncAnimation(
        fig, update, frames=len(predictions_history),
        init_func=init, blit=True, interval=50
    )
    if save_path:
        ani.save(save_path, writer='pillow', fps=20)
        print("儲存完成")
        plt.show()
    else:
        plt.show()