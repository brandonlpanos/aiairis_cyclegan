import os
import matplotlib.pyplot as plt

# Remove .DS_Store files in the base directory and its subdirectories
base_dir = "../data"  # Base directory
def remove_ds_store_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == ".DS_Store":
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Removed .DS_Store file: {file_path}")

def quick_look_gen(real_aia, real_iris, fake_iris, fake_aia, savename=None):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))
    ax1.set_title('AIA (REAL)')
    ax1.imshow(real_aia, aspect='auto', cmap='binary')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_title('IRIS (FAKE)')
    ax2.imshow(fake_iris, aspect='auto', cmap='binary')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax3.set_title('IRIS (REAL)')
    ax3.imshow(real_iris, aspect='auto', cmap='binary')
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax4.set_title('AIA (FAKE)')
    ax4.imshow(fake_aia, aspect='auto', cmap='binary')
    ax4.set_xticks([])
    ax4.set_yticks([])
    plt.tight_layout()
    if savename is not None:
        plt.savefig(f'../callbacks/pics/{savename}.png', bbox_inches='tight')
    plt.close(fig)
    return None