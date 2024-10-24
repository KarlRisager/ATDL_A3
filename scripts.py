

def train_model(num_epochs, model, data, optimizer, criterion, return_loss = True):
    loss_list = []
    print("Training model...\n")
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        y_pred = model(data.x, data.edge_index)
        loss = criterion(y_pred[data.train_mask], data.y[data.train_mask])
        if return_loss:
            loss_list.append(loss.item())
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}", end="\r", flush=True)
    if return_loss:
        return loss_list


def test_model(mask, model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    correct = pred[mask] == data.y[mask]  # Check against ground-truth labels.
    acc = int(correct.sum()) / int(mask.sum())  # Derive ratio of correct predictions.
    return acc