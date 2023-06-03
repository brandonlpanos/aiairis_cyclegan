import torch
import config
from utils import quick_look_gen
import torch.nn as nn
from generator import Generator
from dataset import create_loaders
from discriminator import Discriminator
import torch.optim.lr_scheduler as lr_scheduler
'''
We make use of autocast() and GradScaler() to speed up training and improve stability.
Autocast() allows us to use mixed precision training, i.e., using both FP32 and FP16 operations.
Autocast() matches the dtype to the operation (e.g., FP16 for matrix multiplication and FP32 for addition, etc.)
GradScaler() scales the loss to prevent underflow and overflow.
The Generators are ResNet-based (9 residual blocks) and the Discriminators are PatchGANs (70x70).
The (70X70) PatchGANs are used to classify whether 70x70 overlapping patches in an image are real or fake.
When passing an image to a discriminator, the discriminator outputs a (n x m) matrix, where each value represents the probability that the patch is real. if a value in the matrix is a 1, this means that the discriminator has judged that the corrsponding (70X70) patch from the input is real. 
'''
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DataLoaders
train_loader, test_loader = create_loaders(test_percent=0.2, batch_size=1, sdo_channels=['171'], iris_channel='1400')
# Load Discriminators and Generators
aia_generator = Generator(num_res_blocks=9).to(device)
iris_generator = Generator(num_res_blocks=9).to(device)
aia_discriminator = Discriminator().to(device)
iris_discriminator = Discriminator().to(device)
# Losses
adversarial_criterion = nn.MSELoss()
cycle_criterion = nn.L1Loss()
# Optimizers
gen_optimizer = torch.optim.Adam(list(aia_generator.parameters(
)) + list(iris_generator.parameters()), lr=0.0002, betas=(0.5, 0.999))
disc_optimizer = torch.optim.Adam(list(aia_discriminator.parameters(
)) + list(iris_discriminator.parameters()), lr=0.0002, betas=(0.5, 0.999))
# Use gradient scaling for mixed precision to prevent underflow and improve stability
# Each parameter’s gradient (.grad attribute) should be unscaled before the optimizer updates the parameters, so the scale factor does not interfere with the learning rate (From the PyTorch doumentation)
generator_scalar = torch.cuda.amp.GradScaler()
discriminator_scalar = torch.cuda.amp.GradScaler()
# Learning rate schedulers should start at 0.0002 and remain constant for the first 100 epochs, and then linearly decay to 0 over the next 100 epochs according to the paper
generator_scheduler = lr_scheduler.LambdaLR(
    gen_optimizer, lambda epoch: 1.0 if epoch < 100 else max(0.0, (total_epochs - epoch) / 100.0))
discriminator_scheduler = lr_scheduler.LambdaLR(
    disc_optimizer, lambda epoch: 1.0 if epoch < 100 else max(0.0, (total_epochs - epoch) / 100.0))
# Training loop
best_loss = 1e9
total_epochs = config.NUM_EPOCHS
for epoch in range(total_epochs):
    print(f"Epoch {epoch+1} of {total_epochs}")
    for batch, (aia, iris) in enumerate(train_loader):
        aia = aia.to(device)
        iris = iris.to(device)
        # Train Discriminators
        with torch.cuda.amp.autocast():
            fake_iris = iris_generator(aia)
            d_iris_fake = iris_discriminator(fake_iris)
            d_iris_real = iris_discriminator(iris.detach())
            d_iris_fake_loss = adversarial_criterion(
                d_iris_fake, torch.zeros_like(d_iris_fake))
            d_iris_real_loss = adversarial_criterion(
                d_iris_real, torch.ones_like(d_iris_real))
            iris_discriminator_loss = (d_iris_fake_loss + d_iris_real_loss)
            fake_aia = aia_generator(iris)
            d_aia_fake = aia_discriminator(fake_aia)
            d_aia_real = aia_discriminator(aia.detach())
            d_aia_fake_loss = adversarial_criterion(
                d_aia_fake, torch.zeros_like(d_aia_fake))
            d_aia_real_loss = adversarial_criterion(
                d_aia_real, torch.ones_like(d_aia_real))
            aia_discriminator_loss = (d_aia_fake_loss + d_aia_real_loss)
            discriminator_loss = (
                aia_discriminator_loss + iris_discriminator_loss) / 2
        disc_optimizer.zero_grad()
        discriminator_scalar.scale(discriminator_loss).backward(retain_graph=True)
        discriminator_scalar.step(disc_optimizer)
        discriminator_scalar.update()
        # Train Generators
        with torch.cuda.amp.autocast():
            # Adversarial loss for both generators
            d_iris_fake = iris_discriminator(fake_iris)
            d_aia_fake = aia_discriminator(fake_aia)
            iris_generator_loss = adversarial_criterion(
                d_iris_fake, torch.ones_like(d_iris_fake))
            aia_generator_loss = adversarial_criterion(
                d_aia_fake, torch.ones_like(d_aia_fake))
            # Cycle loss
            cycle_iris = iris_generator(fake_aia)
            cycle_aia = aia_generator(fake_iris)
            cycle_iris_loss = cycle_criterion(iris, cycle_iris)
            cycle_aia_loss = cycle_criterion(aia, cycle_aia)
            # Total generator loss
            generator_loss = (aia_generator_loss +
                              iris_generator_loss +
                              cycle_iris_loss*config.LAMBDA_CYCLE +
                              cycle_aia_loss*config.LAMBDA_CYCLE)
        gen_optimizer.zero_grad()
        generator_scalar.scale(generator_loss).backward(retain_graph=True)
        generator_scalar.step(gen_optimizer)
        generator_scalar.update()
        # Print loss and Plot results 
        if batch % 2 == 0:
            print(f"Generator loss: {generator_loss:.4f}, Discriminator loss: {discriminator_loss:.4f}")
            with torch.no_grad():
                real_iris, real_aia = next(iter(test_loader))
                fake_iris = iris_generator(aia)
                fake_aia = aia_generator(iris)
                real_iris = real_iris.detach().numpy().squeeze()
                real_aia = real_aia.detach().numpy().squeeze()
                fake_iris = fake_iris.detach().numpy().squeeze()
                fake_aia = fake_aia.detach().numpy().squeeze()
                quick_look_gen(real_aia, real_iris, fake_iris, fake_aia, savename=str(batch))
    # Update learning rates
    generator_scheduler.step()
    discriminator_scheduler.step()
    # Save models
    if generator_loss < best_loss:
        best_loss = generator_loss
        torch.save(aia_generator.state_dict(), '../callbacks/models/aia_generator.pth')
        torch.save(iris_generator.state_dict(), '../callbacks/models/iris_generator.pth')
        torch.save(aia_discriminator.state_dict(), '../callbacks/models/aia_discriminator.pth')
        torch.save(iris_discriminator.state_dict(), '../callbacks/models/iris_discriminator.pth')