def plot_grid(grid, start, goal, path=None):
    rows, cols = len(grid), len(grid[0])
    img = np.array(grid, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, cmap=ListedColormap(['white', 'black']), origin='lower',
              extent=[-0.5, cols - 0.5, -0.5, rows - 0.5])

    
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which='minor', color='lightgray', linewidth=0.5)
    ax.tick_params(which='minor', length=0)
    ax.set_xticks(range(cols)); ax.set_yticks(range(rows))
    ax.tick_params(axis='both', which='major', labelsize=8)

    ax.plot(*start, 'o', ms=14, mfc='lime', mec='black', label='Start')
    ax.plot(*goal,  'o', ms=14, mfc='red',  mec='black', label='Goal')

    if path:
        xs = [p[0] for p in path]; ys = [p[1] for p in path]
        ax.plot(xs, ys, '-', color='dodgerblue', lw=3, label='Path')
        ax.plot(xs, ys, 'o', color='dodgerblue', ms=4)

    ax.legend(loc='lower right')
    ax.set_xlim(-0.5, cols - 0.5); ax.set_ylim(-0.5, rows - 0.5)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()