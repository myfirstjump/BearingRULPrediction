import numpy as np
import matplotlib.pyplot as plt

# 生成一個示例信號，例如一個正弦波
# 設定信號參數
fs = 1000  # 采樣率（每秒樣本數）
T = 1.0    # 信號總持續時間（秒）
N = int(T * fs)  # 總樣本數
t = np.linspace(0.0, T, N, endpoint=False)  # 生成時間序列
f = 50.0  # 信號頻率（Hz）
x = np.sin(2 * np.pi * f * t)  # 生成正弦波信號

# 執行傅立葉轉換
X = np.fft.fft(x)

# 計算對應的頻率
freqs = np.fft.fftfreq(N, 1/fs)

# 繪製原始信號
plt.subplot(2, 1, 1)
plt.plot(t, x)
plt.title('Original Signal')

# 繪製頻譜（頻率域表示）
plt.subplot(2, 1, 2)
plt.plot(freqs, np.abs(X))
plt.title('Frequency Domain')
plt.xlabel('Frequency (Hz)')
plt.tight_layout()

plt.show()