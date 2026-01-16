import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse

from data_gen import FunctionGen
from mod import MLP
from visualizer import animate_training

def parse_args():
    parser = argparse.ArgumentParser(description="MLP Function Approximation")
    
    parser.add_argument('--hidden', type=int, default=128, help='隱藏層神經元數量')
    parser.add_argument('--lr', type=float, default=0.002, help='學習率')
    parser.add_argument('--epochs', type=int, default=3000, help='訓練回合數')
    parser.add_argument('--points', type=int, default=80, help='訓練數據點數量')

    parser.add_argument('--activation', type=str, default='tanh', choices=['tanh', 'relu', 'sigmoid', 'leakyrelu', 'gelu'], help='激活函數')
    
    parser.add_argument('--func', type=str, default=None, 
                        help='自定義數學函數')
    
    return parser.parse_args()

def main():
    args = parse_args()

    HIDDEN_SIZE = args.hidden
    LEARNING_RATE = args.lr
    EPOCHS = args.epochs
    CUSTOM_FUNC = args.func

    ACTIVATION = args.activation

    NUM_POINTS = args.points      
    SNAPSHOT_STEP = 20    

    print("="*40)
    print(f"配置參數:")
    print(f"  - Hidden Size: {HIDDEN_SIZE}")
    print(f"  - Learning Rate: {LEARNING_RATE}")
    print(f"  - Epochs: {EPOCHS}")
    if CUSTOM_FUNC:
        print(f"  - 目標函數: f(x) = {CUSTOM_FUNC}")
    else:
        print(f"  - 目標函數: 隨機生成")
    print("="*40)

    try:
        gen = FunctionGen(inF=CUSTOM_FUNC)
        x_dense, y_dense, x_train_tensor, y_train_tensor = gen.get_data(num_points=NUM_POINTS)
    except Exception as e:
        print(f"\n[錯誤] 無法生成數據: {e}")
        print("請檢查函數表達式是否正確")
        return

    model = MLP(hidden_size=HIDDEN_SIZE, activation=ACTIVATION)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    predictions_history = []

    print(f"開始訓練")
    for epoch in range(EPOCHS):
        model.train()
        
        output = model(x_train_tensor)
        loss = criterion(output, y_train_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch < 50 or epoch % SNAPSHOT_STEP == 0 or epoch == EPOCHS - 1:
            model.eval() 
            with torch.no_grad():
                pred = model(x_train_tensor).cpu().numpy()
                predictions_history.append(pred)
            
            if epoch % 500 == 0:
                print(f"Epoch [{epoch}/{EPOCHS}], Loss: {loss.item():.4f}")

    print("訓練完成")

    x_train_np = x_train_tensor.numpy()
    y_train_np = y_train_tensor.numpy()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = "Data Fitting.gif"
    save_full_path = os.path.join(current_dir, file_name)

    animate_training(
        x_dense, y_dense, 
        x_train_np, y_train_np, 
        predictions_history,
        save_path=save_full_path 
    )

if __name__ == "__main__":
    main()
    print("完成")