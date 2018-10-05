import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import ScalarFormatter
from pandas.tseries.offsets import Day
from IPython.core.debugger import set_trace

sns.set_style('ticks')
#mpl.rc('font', family='NanumGothic')
mpl.rc('axes', unicode_minus=False)
plt.rcParams['mathtext.fontset'] = 'cm'

class Plotter(object):

    @classmethod
    def plot_contr_cum(cls, contr, assets=None):
        if assets is None:
            contr_cum = contr.add(1, fill_value=0).cumprod()
        else: 
            contr_cum = contr[assets].add(1, fill_value=0).cumprod()

        contr_cum.plot(figsize=(20,10))


    @classmethod
    def plot_cum(cls, prices, strats, names=None, color=None, style=None, logy=True, start=None, end=None):
        prices_ = prices[strats]
        
        if start is not None:
            prices_ = prices_.loc[start:]
            prices_ /= prices_.iloc[0]
            
        if end is not None:
            prices_ = prices_.loc[:end]
        
        ax = prices_.plot(
            figsize=(7,5), 
            logy=logy, color=color, style=style, 
            xlim=(prices_.index[0], prices_.index[-1]), 
        )

        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel('')
        ax.set_title('Cumulative Return', fontsize=15, weight='bold')

        legend_fsize = 12
        if names: ax.legend(names, fontsize=legend_fsize)
        else: ax.legend(fontsize=legend_fsize)

            
    @classmethod
    def plot_cum_te(cls, cum, strats, bm, te_target, names=None, color=None, logy=True):
        fig, axes = plt.subplots(1, 2, sharey=False, sharex=True, figsize=(10,4))
        cum_ = cum[strats]
        cum_.plot(color=color, legend=False, ax=axes[0], logy=logy)

        rtns = cum_.pct_change()
        #set_trace()
        rtns_ = rtns.drop([bm], axis=1)
        rtns_ = rtns_.sub(rtns[bm], axis=0).rolling(250).std()*(250**0.5)
        rtns_.plot(color=color[1:], legend=False, ax=axes[1])

        axes[0].set_title('Cumulative Return', fontsize=15, weight='bold')
        axes[1].set_title('Tracking error', fontsize=15, weight='bold')
        axes[1].axhline(te_target, color='k', linestyle='--', linewidth=1)

        axes[0].legend(
            names, 
            bbox_to_anchor=(0, 1.1, 1, 0), ncol=4, loc=3
        );
        

    @classmethod
    def plot_cum_exc_te(cls, cum, strats, bm, te_target, names=None, color=None, logy=True):
        fig, axes = plt.subplots(1, 3, sharey=False, sharex=True, figsize=(10.5,3.5))
        cum_ = cum[strats]
        cum_.plot(color=color, legend=False, ax=axes[0], logy=logy)
        #set_trace()
        cum_.sub(cum_[bm], axis=0).drop([bm], axis=1).plot(legend=False, ax=axes[1], color=color[1:], logy=logy)
        axes[1].axhline(0, color='k', linestyle='-', linewidth=1)
        
        rtns = cum_.pct_change()
        #set_trace()
        rtns_ = rtns.drop([bm], axis=1)
        rtns_ = rtns_.sub(rtns[bm], axis=0).rolling(250).std()*(250**0.5)
        rtns_.plot(color=color[1:], legend=False, ax=axes[2])

        axes[0].set_title('Cumulative Return', fontsize=15, weight='bold')
        axes[1].set_title('Relative to ' + names[0], fontsize=15, weight='bold')
        axes[2].set_title('Tracking error', fontsize=15, weight='bold')
        axes[2].axhline(te_target, color='k', linestyle='--', linewidth=1)

        axes[0].legend(
            names, 
            bbox_to_anchor=(0, 1.1, 1, 0), ncol=4, loc=3
        );
        
            
    @classmethod
    def plot_cum_te_many(cls, cum, strats, bm, te_target_list, etas, names=None, color=None):
        fig, axes = plt.subplots(3, len(etas), sharey='row', sharex=True, figsize=(len(etas)*2,3*2));
        strats_ = strats.copy()
        strats_.remove(bm)
        
        for i,strat in enumerate(strats_):
            title = 'TE<'+str(int(te_target_list[i]*100))+'%' if te_target_list[i] else 'No Constraint'
            #set_trace()
            cum_ = cum[[strats_[i],bm]]
            #set_trace()
            cum_.plot(ax=axes[0,i], legend=False, color=color, title=title, xticks=cum_.index[::1250])
            
            rtns = cum_.pct_change()
            ((rtns[strats_[i]]-rtns[bm]).rolling(250).std()*(250**0.5)).plot(ax=axes[1,i], legend=False, color='k')
            if te_target_list[i]: axes[1,i].axhline(te_target_list[i], color='k', linestyle='--', linewidth=1)
            
            etas[i].plot.area(ax=axes[2,i], legend=False, color='silver', ylim=(0,1))
        
        axes[0,0].set_ylabel('Cumulative\n Return')
        axes[1,0].set_ylabel('Tracking error')
        axes[2,0].set_ylabel('Total weight ($\eta$)')
        
        axes[0,0].legend(
            names, 
            bbox_to_anchor=(0, 1.15, 1, 0), ncol=2, loc=3
        );



    @classmethod
    def plot_cum_multi_periods(cls, cum, names=None, color=None, style=None, logy=True, separator=[]):
        #years = cum.index.year.unique()
        #eoy = None
        
        cum_list = []
        start = cum.index[0]
        separator.append(cum.index[-1])

        for isep, sep in enumerate(separator):
            cum_ = cum.loc[start:sep]

            if len(cum_)>1:
                cum_ /= cum_.iloc[0]
                cum_list.append((start, cum_))

            start = cum_.index[-1]

        nFig = len(cum_list)
        nWidth = len(separator)
        nHeight = 1 #int(np.ceil(float(nFig)/nWidth))
        fSize_w = 5
        fSize_h = 4

        fig, axes = plt.subplots(nHeight, nWidth, sharey=True, figsize=(fSize_w*nWidth, fSize_h*nHeight))
        axes = axes.flatten()
        plt.subplots_adjust(hspace=0.6)
        [ax.axis('off') for ax in axes]

        for i, (start, cum_) in enumerate(cum_list):
            ax = axes[i]
            ax.axis('on')
            cum_.plot(ax=ax, legend=False, logy=logy, color=color, style=style, xlim=(cum_.index[0],cum_.index[-1]))#, xticks=cum_.index[::250])
            ax.set_title('start: '+str(start.date()), fontsize=15, weight='bold')
            #ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))

        if names is None:
            names = cum.columns #strats

        axes[0].legend(names, bbox_to_anchor=(0, 1.2, nWidth, 0), ncol=len(names), loc=3);
        
        

    @classmethod
    def plot_cum_yearly(cls, cum, names=None, color=None, style=None, remove=[]):
        years = cum.index.year.unique()
        eoy = None
        cum_list = []

        for iyear, year in enumerate(years):
            cum_ = cum.loc[eoy:str(year)]

            if (len(cum_)>1) and (year not in remove):
                cum_ /= cum_.iloc[0]
                cum_list.append((year, cum_))

            eoy = cum_.index[-1]


        nFig = len(cum_list)
        nWidth = 5
        nHeight = int(np.ceil(float(nFig)/nWidth))
        fSize = 2.5

        fig, axes = plt.subplots(nHeight, nWidth, sharey=True, figsize=(fSize*nWidth, fSize*nHeight))
        axes = axes.flatten()
        plt.subplots_adjust(hspace=0.6)
        [ax.axis('off') for ax in axes]

        for i, (year, cum_) in enumerate(cum_list):
            ax = axes[i]
            ax.axis('on')
            cum_.plot(ax=ax, legend=False, xticks=cum_.index[::60], color=color, style=style, xlim=(cum_.index[0],cum_.index[-1]))
            ax.set_title(year, fontsize=15, weight='bold')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))

        if names is None:
            names = cum.columns #strats

        axes[0].legend(names, bbox_to_anchor=(0, 1.2, nWidth, 0), ncol=len(names), loc=3);


    @classmethod
    def plot_turnover(cls, turnover):    
        ax = turnover.plot(ylim=(0,10), color='k', xlim=(turnover.index[0], turnover.index[-1]))
        ax.set_title('Turnover ratio (12M)', fontsize=15, weight='bold')
        ax.axhline(turnover.mean(), color='k', linestyle='--', linewidth=1);


    @classmethod
    def plot_breakdown(cls, model_contr, weight):
        contr = model_contr.mean()
        contr /= contr.sum()
        p_break = pd.DataFrame()
        p_break['contr'] = contr*100
        p_break['n_month'] = (weight!=0).sum()
        p_break = p_break.sort_values(by=['contr'])

        ax = p_break.plot.barh(
                subplots=True, legend=False, sharex=False, sharey=True, width=0.8,
                figsize=(7, len(p_break)/3.0), 
                layout=(1, 2), 
                #color=('k', 'k'), 
                edgecolor='k', 
                lw=1, 
        )

        ax[0,0].set_title('Contribution (Total=100)', fontsize=15, weight='bold')
        ax[0,1].set_title('# of months', fontsize=15, weight='bold')
        ax[0,0].axvline(0, color='k', linestyle='-', linewidth=1)  


    @classmethod
    def plot_weight(cls, weight, rng, supporter, cash_equiv):
        weight_ = weight.copy()#.drop([cash_equiv], axis=1)
        weight_i = weight_.index + 5*Day()
        weight_.index = weight_i
        weight_ = weight_[str(rng[0]):str(rng[1])]
        #weight_ = weight.loc[str(rng[0]):str(rng[1])]
        weight_.index = weight_.index.strftime('%Y-%m')
        
        weight__ = []
        for dt in weight_.index:
            has_weight = weight_.loc[dt].abs() > 0.001
            weight__.append(weight_.loc[dt][has_weight])

        #set_trace()
        weight__ = pd.DataFrame(weight__)
        cols = list(weight__.columns)
        if supporter in cols: cols.remove(supporter)
        if cash_equiv in cols: cols.remove(cash_equiv)
        if supporter==cash_equiv: 
            cols = [cash_equiv] + cols
        else:
            cols = [cash_equiv, supporter] + cols
            
        #weight__ = weight__[cols].drop([cash_equiv], axis=1)
        weight__ = weight__.reindex(columns=cols).drop([cash_equiv], axis=1)

        bar_w = 0.8
        fig_h = len(weight__)/3.0
        ax = weight__.plot.barh(stacked=True, figsize=(10,fig_h), colormap='tab20c', width=bar_w, xlim=(0,1))
        ax.legend(loc=1, bbox_to_anchor=(1.25, 1));


    @classmethod
    def plot_stats(cls, stats, strats, items, names=None, color=None, lim=None, ncols=3, hbar=0.6, hspace=0.7):
        height_strats = hbar # 전략별 bar 높이
        n_items = len(items)
        n_cols = ncols
        n_rows = int(np.ceil(n_items/float(n_cols)))
        fig_width = ncols * 8.0/3.0 #8

        stats_ = stats.loc[strats, items.keys()]#.copy()
        if names: stats_.index = names
        if color: color = [color] * n_items

            
        fig, ax = plt.subplots(n_rows, n_cols, figsize=(fig_width, height_strats*len(strats)*n_rows), sharex=False, sharey=True);
        
        stats_.plot.barh(
            subplots=True, legend=False, width=0.8, #sharex=False, sharey=True, 
            #figsize=(fig_width, height_strats*len(strats)*n_rows), 
            #layout=(n_rows, n_cols), 
            title=items.values(), 
            color=color, 
            edgecolor='k', 
            lw=1,
            #xerr=err_value, 
            ax=ax, 
        )

        for i, ax_ in enumerate(ax.flatten()):
            if lim: ax_.set_xlim(lim[i])
            ax_.axvline(0, color='k', linestyle='-', linewidth=1)

        plt.subplots_adjust(hspace=hspace)
        #fig.suptitle('Statistics', fontsize=15, weight='bold', y=0.94)


    @classmethod
    def plot_profile(cls, stats, strats, names=None, color=None, bsize=None):
        cagr = stats['cagr']
        std = stats['std']

        # 차트범위 최대값
        lim = np.ceil(max(cagr.max(), std.max()) * 1.1 / 5) * 5

        # 듀얼모멘텀을 지나는 직선들
        x0, y0 = std[strats], cagr[strats]
        slope = y0 / x0
        X_ = np.linspace(0, lim, 100)
        Y_ = slope.values * X_.reshape(-1,1)
        ax = pd.DataFrame(Y_, index=X_).plot(zorder=-1, style='k-', legend=False)

        # 위험조정수익률=1 인 직선
        pd.Series(X_, index=X_).plot(zorder=-1, style='k--', legend=False, ax=ax)

        # color 설정
        i_strats = stats.index.get_indexer(strats)
        c_ = np.full(len(std), None)
        c_[:] = 'k'
        if color: c_[i_strats] = color

        # 버블 사이즈 설정
        s_ = np.full(len(std), None)
        s_[:] = 100
        if bsize: s_[i_strats] = bsize

        # 라벨 설정
        labels = stats.index.values.copy() # copy안하면 원래 index가 바뀌어버린다
        if names: labels[i_strats] = names

        # Scatter plot
        stats.plot.scatter(
            x='std', y='cagr', ax=ax, edgecolor='k', 
            xlim=(0,lim), ylim=(0,lim), figsize=(7,7), 
            s=s_.tolist(), 
            c=c_.tolist(), 
            lw=1,
        )

        # Annotation
        for label, x, y in zip(labels, std, cagr):
            ax.annotate(
                label, 
                xy=(x,y), 
                xytext=(5,5),
                textcoords='offset points', 
                ha='left', #'right', 
                va='bottom',
                bbox=dict(facecolor='w', alpha=0.8, lw=1), 
                size=12,
            )

        ax.set_xlabel('Volatility (%)', size=15) # 연변동성
        ax.set_ylabel('CAGR (%)', size=15)


    @classmethod
    def plot_dist(cls, prices, strats, items, n_roll_stats=250, names=None, color=None):
        height_strats = 1.5
        prices_ = prices[strats]#.copy()
        if names: prices_.columns = names

        fig, axes = plt.subplots(len(strats), len(items), figsize=(11,height_strats*len(strats)))
        prices_rolled = prices_.rolling(n_roll_stats)

        for i, (item_, label_) in enumerate(items.items()):
            collected = prices_rolled.apply(item_)
            med = collected.median()
            legend = True if i==0 else False

            ax = collected.plot.hist(
                bins=50, edgecolor='k', subplots=True, 
                sharex=True, histtype='stepfilled', 
                color=color, 
                ax=axes[:,i], 
                legend=legend, 
                lw=1,
            )

            for j, ax_ in enumerate(ax):
                ax_.axvline(0, color='k', linestyle='--', linewidth=1)
                ax_.axvline(med[j], color='r', linewidth=5, alpha=0.5)
                ax_.set_ylabel('')

            #ax[0].set_title(label_, size=15)
            ax[-1].set_xlabel(label_, size=15)

            

    @classmethod
    def plot_stats_pool(cls, stats_pool, items, names=None, lim=None):
        stats_pool_ = stats_pool.loc[:,items.keys()]#.sort_index()
        if names: stats_pool_.index = names

        f_height = 1.5
        ax = stats_pool_.plot.bar(
            subplots=True, sharex=True, sharey=False, legend=False, 
            width=0.8, color='k', 
            layout=(len(items),1), 
            figsize=(5,f_height*len(items)), 
            title=items.values(), 
        )

        for i, ax_ in enumerate(ax): 
            if lim: ax[i,0].set_ylim(lim[i])
            #ax[i,0].set_title(fontsize=15, weight='bold')

        plt.subplots_adjust(hspace=0.5)
        
        
        
def plot_normal_dist_under0_shaded():
    mu = 1
    std = 1
    rv = stats.norm(mu, std)

    x = np.linspace(-5, 5, 1000)
    y = rv.pdf(x)
    plt.plot(x, y, color='k', lw=2)
    plt.xlim(-3, 5)
    plt.ylim(0, 0.5)

    plt.axvline(0, color='k', linestyle='-', linewidth=1)  
    plt.axhline(0, color='k', linestyle='-', linewidth=1)
    plt.fill_between(x[x<=0], y[x<=0], 0, color='silver')
    
    
    
def plot_kmeans(mode):
    if mode==0:
        X = np.array([[1,1],[2,1],[4,3],[5,4]])

        fig = plt.figure(figsize=(3,3))
        plt.scatter(X[:,0],X[:,1], color='k')
        plt.ylim(0,6)
        plt.xlim(0,6)

        plt.text(0.8, 1.5, '$\mathbf{x}_1$', fontsize=20, color='k')
        plt.text(1.8, 1.5, '$\mathbf{x}_2$', fontsize=20, color='k')
        plt.text(3.8, 3.5, '$\mathbf{x}_3$', fontsize=20, color='k')
        plt.text(4.8, 4.5, '$\mathbf{x}_4$', fontsize=20, color='k')
        
    elif mode==1:
        X = np.array([[1,1],[2,1],[4,3],[5,4]])
        x = np.linspace(0, 6, 100)

        fig, axes = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(9,3));

        axes[0].axvline(1.5, color='k', linestyle='--', lw=1)
        axes[1].plot(x,-(2.7)/(1.7)*(x-(4.7)/2)+(3.7)/2, '--k', lw=1)
        axes[2].plot(x,-(3)/(2.5)*(x-(6)/2)+(4.5)/2, '--k', lw=1)

        axes[0].scatter(1,1, color='r', s=400)
        axes[0].scatter(2,1, color='gold', s=400)
        axes[0].set_title('Iteration 0')

        axes[1].scatter(1,1, color='r', s=400)
        axes[1].scatter(3.7, 2.7, color='gold', s=400)
        axes[1].set_title('Iteration 1')

        axes[2].scatter(1.5,1, color='r', s=400)
        axes[2].scatter(4.5, 3.5, color='gold', s=400)
        axes[2].set_title('Iteration 2')

        axes[0].scatter(X[:,0],X[:,1], color='k')
        axes[1].scatter(X[:,0],X[:,1], color='k')
        axes[2].scatter(X[:,0],X[:,1], color='k')
        plt.ylim(0,6)
        plt.xlim(0,6)
        
        
def plot_elasticity():
    fig = plt.figure(figsize=(8,4))
    x = np.linspace(0, 3, 100)

    plt.subplot(121)
    plt.plot(x, x**0.2, 'k')
    plt.plot(x, x**1, 'b')
    plt.plot(x, x**2, 'r')

    plt.title(r'$\alpha > 0$', fontsize=20, y=1.03)
    plt.xlim(0,4)
    plt.ylim(0,5.5)
    plt.text(3.2, 1.1, '$\mathbf{R}_i^{0.2}$', fontsize=15, color='k')
    plt.text(3.2, 3, '$\mathbf{R}_i^1$', fontsize=15, color='b')
    plt.text(2.5, 5, '$\mathbf{R}_i^2}$', fontsize=15, color='r')

    x = np.linspace(0.001, 3, 1000)

    plt.subplot(122)
    plt.plot(x, x**(-0.2), 'k')
    plt.plot(x, x**(-1), 'b')
    plt.plot(x, x**(-2), 'r')

    plt.text(2, 1.1, '$\mathbf{R}_i^{-0.2}$', fontsize=15, color='k')
    plt.text(3.2, 0.2, '$\mathbf{R}_i^{-1}$', fontsize=15, color='b')
    plt.text(0.5, 5, '$\mathbf{R}_i^{-2}}$', fontsize=15, color='r')

    plt.title(r'$\alpha < 0$', fontsize=20, y=1.03)
    plt.xlim(0,4)
    plt.ylim(0,5.5)
    
    
    
def plot_max_single_weight(*bts, names=None, figwidth=10):
    fig, axes = plt.subplots(1, len(bts), figsize=(figwidth,3))
    fig.suptitle('History of Maximum single weight', fontsize=15, weight='bold', y=1.05)
    
    for i,bt in enumerate(bts):
        bt.weight.max(axis=1).plot.area(ylim=(0,1), color='silver', ax=axes[i])
        axes[i].axhline(0.5, color='k', linestyle='--', linewidth=1)
        if names: axes[i].set_title(names[i])
    
    
    
def plot_heat(bt_pool, slotx, sloty, items=['cagr', 'sharpe'], names=['CAGR','Sharpe'], labels=[r'$\beta$',r'$\alpha$     ']):
    #slots = np.linspace(0,2,11)

    tb_0 = np.array([bt_pool.backtests[bt].stats.loc['DualMomentum',items[0]] for bt in bt_pool.backtests]).reshape(11,11)
    tb_0 = pd.DataFrame(tb_0, index=sloty, columns=slotx).sort_index(ascending=False)

    tb_1 = np.array([bt_pool.backtests[bt].stats.loc['DualMomentum',items[1]] for bt in bt_pool.backtests]).reshape(11,11)
    tb_1 = pd.DataFrame(tb_1, index=sloty, columns=slotx).sort_index(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(12,6))

    sns.heatmap(tb_0, annot=True, square=True, fmt='.1f', linewidths=.1, cbar=False, cmap='YlGnBu', ax=axes[0])
    axes[0].set_title(names[0], size=20)
    axes[0].set_ylabel(labels[1], size=20, rotation=0)
    axes[0].set_xlabel(labels[0], size=20)

    sns.heatmap(tb_1, annot=True, square=True, fmt='.2f', linewidths=.1, cbar=False, cmap='YlGnBu', ax=axes[1])
    axes[1].set_title(names[1], size=20)
    axes[1].set_ylabel(labels[1], size=20, rotation=0)
    axes[1].set_xlabel(labels[0], size=20)    
    
    
    
def plot_by_eaa_bnd(ref, *bts, names=['Dual momentum', 'Dynamic EAA w/o CP']):
    fig, axes = plt.subplots(3, len(bts), sharey='row', sharex=True, figsize=(len(bts)*2,3*2));

    for i,bt in enumerate(bts):
        ref.cum[['DualMomentum']].plot(ax=axes[0,i], legend=False, color='orange', xticks=bt.cum.index[::1250])
        bt.cum[['DualMomentum']].plot(ax=axes[0,i], legend=False, color='r', title=r'$\theta$='+str(i+1))
        pd.Series(bt.port.wr, index=bt.dates_asof).plot(ax=axes[1,i], lw=1, ylim=(-6,6))
        bt.weight.max(axis=1).plot.area(color='silver', ax=axes[2,i], ylim=(0,1))
    
    axes[0,0].set_ylabel('Cumulative\n Return')
    axes[1,0].set_ylabel(r'Elasticity $\alpha$')
    axes[2,0].set_ylabel('Max\n single weight')

    axes[0,0].legend(
        names, 
        bbox_to_anchor=(0, 1.15, 1, 0), ncol=2, loc=3
    );    
    
    

def plot_te_filter(what):    
    if what=='base':
        x1 = np.linspace(-2, 0, 100)
        y1 = np.zeros_like(x1)
        x2 = np.linspace(0, 1, 50)
        y2 = np.sqrt(x2)
        x3 = np.linspace(1,2,50)
        y3 = np.ones_like(x3)
        x4 = np.linspace(1,2,50)
        y4 = np.sqrt(x4)
        plt.plot(x1, y1, 'k', lw=5)
        plt.plot(x2, y2, 'k', lw=5)
        plt.plot(x3, y3, 'k', lw=5)
        plt.plot(x4, y4, 'k--', lw=1)
        plt.xlim(-2,2)
        plt.ylim(-0.1,1.2)
        plt.title('$y=\eta^*(x)$', fontsize=20, y=1.03)
    
    elif what=='smoother':
        x1 = np.linspace(-2, 0, 100)
        y1 = np.zeros_like(x1)
        x2 = np.linspace(0, 1, 50)
        y2 = np.sqrt(x2)
        x3 = np.linspace(1,2,50)
        y3 = np.ones_like(x3)
        k = 0.3
        x4 = np.linspace(-2,2,200)
        y4 = np.sqrt(k) * np.exp(0.5*(x4/k)-0.5)

        plt.plot(x4, y4, 'r--', lw=1)
        plt.plot(x1, y1, 'k', lw=5)
        plt.plot(x2, y2, 'k', lw=5)
        plt.plot(x3, y3, 'k', lw=5)

        plt.xlim(-2,2)
        plt.ylim(-0.1,1.2)
        plt.axvline(k, color='k', linestyle='-', linewidth=1)
        plt.text(0.25, -0.22, '$k$', fontsize=20, color='r')
        plt.text(-1, 0.2, '$g(x)$', fontsize=20, color='r')
        plt.text(1, 0.85, '$\eta^*(x)$', fontsize=20, color='k')
        
    elif what=='base+smoother':
        x1 = np.linspace(-2, 0, 100)
        y1 = np.zeros_like(x1)
        x2 = np.linspace(0.3, 1, 50)
        y2 = np.sqrt(x2)
        x3 = np.linspace(1,2,50)
        y3 = np.ones_like(x3)
        k = 0.3
        x4 = np.linspace(-2,0.3,200)
        y4 = np.sqrt(k) * np.exp(0.5*(x4/k)-0.5)

        x5 = np.linspace(0, 0.3, 50)
        y5 = np.sqrt(x5)

        plt.plot(x4, y4, 'k', lw=5)
        plt.plot(x1, y1, 'k--', lw=1)
        plt.plot(x2, y2, 'k', lw=5)
        plt.plot(x3, y3, 'k', lw=5)
        plt.plot(x5, y5, 'k--', lw=1)
        plt.xlim(-2,2)
        plt.ylim(-0.1,1.2)
        plt.axvline(k, color='k', linestyle='-', linewidth=1)
        plt.text(0.25, -0.22, '$k$', fontsize=20, color='r')
        plt.text(-1, 0.2, '$g(x)$', fontsize=20, color='k')
        plt.text(1, 0.85, '$\eta^*(x)$', fontsize=20, color='k')
        plt.title('$y=\eta^*_o(x)$', fontsize=20, y=1.03)
        
    elif what=='many_k':
        k = 0.1
        x1 = np.linspace(-2, 0, 100)
        y1 = np.zeros_like(x1)
        x2 = np.linspace(k, 1, 50)
        y2 = np.sqrt(x2)
        x3 = np.linspace(1,2,50)
        y3 = np.ones_like(x3)

        x4 = np.linspace(0, k, 50)
        y4 = np.sqrt(x4)

        x5 = np.linspace(-2,k,200)
        y5 = np.sqrt(k) * np.exp(0.5*(x5/k)-0.5)

        k = 0.5
        x6 = np.linspace(-2,k,200)
        y6 = np.sqrt(k) * np.exp(0.5*(x6/k)-0.5)

        k = 0.9
        x7 = np.linspace(-2,k,200)
        y7 = np.sqrt(k) * np.exp(0.5*(x7/k)-0.5)

        plt.plot(x1, y1, 'k', lw=5)
        plt.plot(x2, y2, 'k', lw=5)
        plt.plot(x3, y3, 'k', lw=5)
        plt.plot(x4, y4, 'k', lw=5)
        plt.plot(x5, y5, 'k--', lw=1)
        plt.plot(x6, y6, 'k--', lw=1)
        plt.plot(x7, y7, 'k--', lw=1)
        plt.xlim(-2,2)
        plt.ylim(-0.1,1.2)
        plt.text(-1.8, 0.32, '$k=0.9$', fontsize=15, color='k')
        plt.text(-1.2, 0.22, '$k=0.5$', fontsize=15, color='k')
        plt.text(-0.7, 0.1, '$k=0.1$', fontsize=15, color='k')
        #plt.title('$y=\eta^*_o(x)$', fontsize=20, y=1.03)