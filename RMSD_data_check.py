import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

def add_metrics(x, y, ax=None, s=0, show_corr=True, show_ubRMSD=True, show_bias=True, show_line_eq=True, 
                corner='top-right', color='black', alpha=0.7, fontsize = 8, vminmax=None):
    """
    Add metrics (correlation, ubRMSD, bias, regression line equation) to a matplotlib axis.
    
    Parameters:
        x (array-like): x-axis data.
        y (array-like): y-axis data.
        ax (matplotlib.axes.Axes): Axis to plot the scatter plot and metrics text.
        s (float): Marker size for the scatter plot.
        show_corr (bool): Whether to show the correlation coefficient (R).
        show_ubRMSD (bool): Whether to show the unbiased RMSD (ubRMSD).
        show_bias (bool): Whether to show the bias.
        show_line_eq (bool): Whether to show the regression line equation.
        corner (str): Corner of the axis to place the metrics ('top-right', 'top-left', 'bottom-right', 'bottom-left').
        color (str): Color of the scatter points and regression line.
        alpha (float): Transparency of the scatter points.
    """
    x = np.array(x) # reference
    y = np.array(y) # prediction
    
    # Remove NaN values
    nan_mask1 = ~np.isnan(x) & ~np.isnan(y)
    nan_mask2 = ~np.isinf(x) & ~np.isinf(y)
    nan_mask = nan_mask1&nan_mask2
    x, y = x[nan_mask], y[nan_mask]

    # Calculate metrics
    corr = np.corrcoef(x, y)[0, 1]
    ubRMSD = ubrmsd(x, y)
    bias_ = np.nanmean(y - x)
    
    # Linear regression
    line = LinearRegression()
    line.fit(x.reshape(-1, 1), y.reshape(-1, 1))
    
    # Prepare the metrics text based on user preference
    metrics = []
    if show_corr:
        metrics.append(r'$R=%.3f$' % corr)
    if show_ubRMSD:
        metrics.append(r'$ubRMSD=%.3f$' % ubRMSD)
    if show_bias:
        metrics.append(r'$bias=%.3f$' % bias_)
    if show_line_eq:
        metrics.append(r'$y = {:.4f}x {:+.3f}$'.format(line.coef_[0, 0], line.intercept_[0]))
        
    # Combine metrics text
    textstr = '\n'.join(metrics)
    
    # Determine text position based on corner argument
    corner_positions = {
        'top-right': (0.95, 0.95),
        'top-left': (0.05, 0.95),
        'bottom-right': (0.95, 0.05),
        'bottom-left': (0.05, 0.05),
    }
    position = corner_positions.get(corner, (0.95, 0.95))  # Default to top-right if invalid corner is given
    if ax==None:
        fig,ax = plt.subplots()
        ax.scatter(x, y, s=s, alpha=alpha, color=color)
    else:
        # Plot scatter points
        ax.scatter(x, y, s=s, alpha=alpha, color=color)

    # Get x-axis limits and extend regression line to the full range
    x_limits = ax.get_xlim()
    if vminmax==None:
        x_pred = np.linspace(x_limits[0], x_limits[1], 100)
    else:
        x_pred = np.linspace(vminmax[0], vminmax[1], 100)
    y_pred = line.predict(x_pred.reshape(-1, 1))
    
    # Plot regression line
    ax.plot(x_pred, y_pred, color=color, alpha=0.8, linewidth=2)
    
    # Add text to the specified corner with the chosen text color
    props = dict(boxstyle='round', facecolor='white', alpha=.9)
    ax.text(position[0], position[1], textstr, transform=ax.transAxes, fontsize=fontsize,
            verticalalignment='top' if 'top' in corner else 'bottom',
            horizontalalignment='right' if 'right' in corner else 'left',
            bbox=props, color=color)
    
    # Set face color
    ax.set_facecolor('w')

def corr(predictions, targets):
    x = predictions
    y = targets
    x1 = np.array(x); y1 = np.array(y);
    x1_nanloc = np.isnan(x1)
    y1_nanloc = np.isnan(y1)
    nanloc = x1_nanloc + y1_nanloc    
    x1 = x1[nanloc==0]
    y1 = y1[nanloc==0]
    rho = np.nanmean((x1 - x1.mean()) * (y1 - y1.mean())) / (x1.std() * y1.std())    
    return rho

def ubrmsd(predictions, targets):
    return np.sqrt(np.nanmean((((predictions - np.nanmean(predictions)) - (targets - np.nanmean(targets))) ** 2)))

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def bias(predictions, targets): 
    return (predictions - targets).mean()

def lineartodb(sar):
    return 10*np.log10(sar)

# ==============================================================================
# check data relationship
# ==============================================================================
processed_data = "/Users/jslee/Downloads/SEN12FLOOD/processed/"

X_trn = np.load(processed_data + 'trn_X_VV_VH_RVI_raw.npy')
Y_trn = np.load(processed_data + 'trn_Y_R_Nir_NDVI_raw.npy')
X_tst = np.load(processed_data + 'tst_X_VV_VH_RVI_raw.npy')
Y_tst = np.load(processed_data + 'tst_Y_R_Nir_NDVI_raw.npy')

df = pd.DataFrame({
    'VV':lineartodb(X_trn[:10,0].flatten()),
    'VH':lineartodb(X_trn[:10,1].flatten()),
    'RVI':X_trn[:10,2].flatten(),
    'R':Y_trn[:10,0].flatten(),
    'Nir':Y_trn[:10,1].flatten(),
    'NDVI':Y_trn[:10,2].flatten(),
}).dropna()

i = -2
plt.imshow(X_trn[i, 1], vmin=0, vmax = .1);plt.colorbar()
plt.imshow(X_trn[i, 0], vmin=0, vmax = 1.);plt.colorbar()
plt.imshow(Y_trn[i, 0], vmin=0, vmax = 1.);plt.colorbar()
plt.imshow(Y_trn[i, 2], vmin=0, vmax = 1.);plt.colorbar()


R_type = 'pearson'
# R_type = 'spearman'
# 상관계수 행렬 변수 저장
corr_matrix = df.corr(R_type)

# 기존 코드
plt.imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1)
plt.colorbar(label= f"{R_type} R")

# 하드코딩된 6 대신 df.columns 길이를 사용하면 컬럼 수가 바뀌어도 알아서 적용됩니다.
plt.xticks(np.arange(len(df.columns)), df.columns)
plt.yticks(np.arange(len(df.columns)), df.columns)

# 💡 격자 안에 라벨(텍스트) 추가
for i in range(len(df.columns)):
    for j in range(len(df.columns)):
        val = corr_matrix.iloc[i, j]
        # 배경색이 짙을 때(절댓값이 클 때)는 흰색, 옅을 때는 검은색 텍스트
        text_color = "white" if abs(val) > 0.6 else "black"
        
        # j가 x축(열), i가 y축(행) 좌표입니다.
        plt.text(j, i, f"{val:.2f}", 
                 ha="center", va="center", color=text_color)

plt.show()

# plt.xticks(np.arange(6), df.columns)
# plt.yticks(np.arange(6), df.columns)

# add_metrics(x =X_trn[0,0].flatten(), y= lineartodb(Y_trn[0,0]).flatten(), s=1)
# add_metrics(x =X_trn[0,1].flatten(), y= lineartodb(Y_trn[0,1]).flatten(), s=1)
# add_metrics(x =X_trn[0,2].flatten(), y= Y_trn[0,2].flatten(), s=1)

# X_trn_0 = X_trn[:, 0].flatten().copy()
# Y_trn_0 = Y_trn[:, 0].flatten().copy()
# len(Y_trn_0)
# len(X_trn_0)

# import seaborn as sns
# sns.histplot(x =X_trn_0,y= Y_trn_0, bins = 50)
# np.corrcoef(X_trn_0, Y_trn_0)
