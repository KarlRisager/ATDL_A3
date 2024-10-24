

def train_model(num_epochs, model, train_loader, optimizer, criterion, device="cpu"):
    for epoch in range(num_epochs):
        for (x,y) in train_loader:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Loss: {loss.item()}")