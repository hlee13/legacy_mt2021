def useStyle(ax):
    # get_xaxis().set_axisline_style("-|>")
    # ['left'] #.set_visible(False) 
    #     .set_position(('outward', 10))
    ax.set_facecolor('#F0F0F0')
    for pos in ['top','left','right','bottom']:
#         ax.spines[pos].set_color('black')
#         ax.spines[pos].set_alpha(0.2)
        ax.spines[pos].set(color='black', alpha=0.2)

    ax.tick_params(direction='out',color='black',length=10)
    ax.grid(linestyle='--', color='black', linewidth=0.1) # ,linewidth=0.3, alpha=0.3
    return ax

@interact
def _(bins=IntSlider(100, 10, 200, 20)):

    plt.style.use('classic')

    fig, ax = plt.subplots(1, 1, figsize=(8,5))
    fig.set_facecolor('#F0F0F0')
    ax = useStyle(ax)
    # ax.set_xlim(left=0,right=10000)
    # ax.set_ylim(bottom=0,top=6)
    s = pd.Series(np.random.randn(2000))

    s.plot(ax=ax, kind='kde', style='.-')
    s.hist(bins=bins, ax=ax, density=True)
    
