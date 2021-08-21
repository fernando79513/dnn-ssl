# Cool 3d graph



if __name__ == "__main__":
    x   = np.array(range(73))
    y1  = np.ones(x.size)
    y2  = np.ones(x.size)*2
    y3  = np.ones(x.size)*3
    y4  = np.ones(x.size)*4
    y5  = np.ones(x.size)*5
    y6  = np.ones(x.size)*6
    y7  = np.ones(x.size)*7

    plt.figure()
    ax = plt.subplot(projection='3d')

    ax.plot(x, y1, cc_list[0])
    ax.plot(x, y2, cc_list[1])
    ax.plot(x, y3, cc_list[2])
    ax.plot(x, y4, cc_list[3])
    ax.plot(x, y5, cc_list[4])
    ax.plot(x, y6, cc_list[5])
    ax.plot(x, y7, cc_list[6])

    ax.add_collection3d(plt.fill_between(x, 0, cc_list[0], alpha=0.3), zs=1, zdir='y')
    ax.add_collection3d(plt.fill_between(x, 0, cc_list[1], alpha=0.3), zs=2, zdir='y')
    ax.add_collection3d(plt.fill_between(x, 0, cc_list[2], alpha=0.3), zs=3, zdir='y')
    ax.add_collection3d(plt.fill_between(x, 0, cc_list[3], alpha=0.3), zs=4, zdir='y')
    ax.add_collection3d(plt.fill_between(x, 0, cc_list[4], alpha=0.3), zs=5, zdir='y')
    ax.add_collection3d(plt.fill_between(x, 0, cc_list[5], alpha=0.3), zs=6, zdir='y')
    ax.add_collection3d(plt.fill_between(x, 0, cc_list[6], alpha=0.3), zs=7, zdir='y')


    plt.show()