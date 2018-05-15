import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import ScalarFormatter

sns.set_style('ticks')
#mpl.rc('font', family='NanumGothic')
mpl.rc('axes', unicode_minus=False)


class Plotter(object):

    @classmethod
    def plot_contr_cum(cls, contr, assets=None):
        if assets is None:
            contr_cum = contr.add(1, fill_value=0).cumprod()
        else: 
            contr_cum = contr[assets].add(1, fill_value=0).cumprod()

        contr_cum.plot(figsize=(20,10))


    @classmethod
    def plot_cum(cls, prices, strats, names=None, color=None, style=None, logy=True):
        prices_ = prices[strats]
        ax = prices_.plot(
            figsize=(7,5), 
            logy=logy, color=color, style=style, 
            xlim=(prices_.index[0], prices_.index[-1]), 
        )

        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel('')

        legend_fsize = 12
        if names: ax.legend(names, fontsize=legend_fsize)
        else: ax.legend(fontsize=legend_fsize)


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
            names = strats

        axes[0].legend(names, bbox_to_anchor=(0, 1.2, nWidth, 0), ncol=len(strats), loc=3);


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
    def plot_weight(cls, weight, rng, riskfree, cash_equiv):
        weight_ = weight.copy().drop([cash_equiv], axis=1)
        weight_i = weight_.index + 5*Day()
        weight_.index = weight_i
        weight_ = weight_[str(rng[0]):str(rng[1])]
        weight_.index = weight_.index.strftime('%Y-%m')

        weight__ = []
        for dt in weight_.index:
            has_weight = weight_.loc[dt].abs() > 0.001
            weight__.append(weight_.loc[dt][has_weight])

        weight__ = pd.DataFrame(weight__)
        cols = list(weight__.columns)
        cols.remove(riskfree)
        cols = [riskfree] + cols
        weight__ = weight__[cols]

        bar_w = 0.8
        fig_h = len(weight__)/3.0
        ax = weight__.plot.barh(stacked=True, figsize=(10,fig_h), colormap='tab20c', width=bar_w, xlim=(0,1))
        ax.legend(loc=1, bbox_to_anchor=(1.25, 1));


    @classmethod
    def plot_stats(cls, stats, strats, items, names=None, color=None, lim=None, ncols=3):
        height_strats = 0.6 # 전략별 bar 높이
        n_items = len(items)
        n_cols = ncols
        n_rows = int(np.ceil(n_items/float(n_cols)))
        fig_width = ncols * 8.0/3.0 #8

        stats_ = stats.loc[strats, items.keys()]#.copy()
        if names: stats_.index = names
        if color: color = [color] * n_items

        ax = stats_.plot.barh(
            subplots=True, legend=False, sharex=False, sharey=True, width=0.8,
            figsize=(fig_width, height_strats*len(strats)*n_rows), 
            layout=(n_rows, n_cols), 
            title=items.values(), 
            color=color, 
            edgecolor='k', 
            lw=1,
            #xerr=err_value, 
        )

        for i, ax_ in enumerate(ax.flatten()):
            if lim: ax_.set_xlim(lim[i])
            ax_.axvline(0, color='k', linestyle='-', linewidth=1)

        plt.subplots_adjust(hspace=0.7)#1.5)


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

        ax.set_xlabel('Standard deviation(%)', size=15) # 연변동성
        ax.set_ylabel('CAGR%)', size=15)


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
