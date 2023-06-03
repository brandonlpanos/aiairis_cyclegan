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

def quick_look_gen(real_aia, fake_iris1, fake_aia1, real_iris, fake_aia2, fake_iris2, savename=None):

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(8, 5))
    ax1.set_title('AIA (REAL)', fontsize=8)
    ax1.imshow(real_aia, aspect='auto', cmap='binary')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_title('IRIS (FAKE)', fontsize=8)
    ax2.imshow(fake_iris1, aspect='auto', cmap='binary')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax3.set_title('AIA (FAKE)', fontsize=8)
    ax3.imshow(fake_aia1, aspect='auto', cmap='binary')
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax4.set_title('IRIS (REAL)', fontsize=8)
    ax4.imshow(real_iris, aspect='auto', cmap='binary')
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax5.set_title('AIA (FAKE)', fontsize=8)
    ax5.imshow(fake_aia2, aspect='auto', cmap='binary')
    ax5.set_xticks([])
    ax5.set_yticks([])
    ax6.set_title('IRIS (FAKE)', fontsize=8)
    ax6.imshow(fake_iris2, aspect='auto', cmap='binary')
    ax6.set_xticks([])
    ax6.set_yticks([])
    plt.tight_layout()

    if savename is not None:
        plt.savefig(f'../callbacks/pics/{savename}.png', bbox_inches='tight')

    plt.close(fig)
    return None