#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    cleaned_data = [(ages[n], net_worths[n], [(predictions[n][0] - net_worths[n][0])**2]) for n in range(len(ages))]
    cleaned_data = sorted(cleaned_data, key = lambda x: x[2])
    cleaned_data = cleaned_data[:int(len(cleaned_data)*.9)]
    
    return cleaned_data

