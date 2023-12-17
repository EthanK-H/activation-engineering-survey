import wandb

def wandb_line_plot(x_values, y_values, name):
    # Create data for the table
    data = [[x, y] for (x, y) in zip(x_values, y_values)]
    
    # Create a wandb.Table and add data to it
    table = wandb.Table(data=data, columns=["x", "y"])
    
    # Log the line plot
    wandb.log({
        name: wandb.plot.line(table, "x", "y", title=name)
    })

def log_bar_chart_to_wandb(data_dict, name):

    # Prepare data for the table
    labels, values = zip(*data_dict.items())
    data = [[label, val] for label, val in zip(labels, values)]

    # Create a wandb.Table and add data to it
    table = wandb.Table(data=data, columns=["label", "value"])

    # Log the bar chart
    wandb.log({
        name: wandb.plot.bar(table, "label", "value", title=name)
    })