import transformers
import numpy as np
from scipy.spatial.distance import cosine

def getSurprisalMetrics(LMOutput: transformers.modeling_outputs.CausalLMOutputWithCrossAttentions, metric: str, transient: bool, drift: float):

    """
    Take hugging face transformer output and return metrics documented in Kumar et. al. 2022.
    
    Parameters
    ---
    LMOutput: transformers.modeling_outputs.CausalLMOutputWithCrossAttentions
        Casual LLM output. Hidden layer output is expected.
    
    metric: str
        Possible options: "WordProbabilities", "WordEntropies", "KLDivergence", "EmbeddingCosines".
        
    transient: bool
        If transient = True, the metrics is normalized to the running average defined by the drift parameter.
        
    drift: float
        Drift defines how much of the surprisal induced by the current token is "remembered" for the calculation of the next token.
    
    Returns
    ---
    metricOutput: list
        Metric per token position.
        
    """
    
    word_probs = softmax(output[0][0], dim = 1).detach().numpy()
    
    if metric == "WordProbabilities":
        metricOutput = np.max(word_probs, axis = 1)
        
    elif metric == "WordEntropies":
        metricOutput = = -np.sum(word_probs * np.log(word_probs), axis = 1)
        
    elif metric == "KLDivergence":
        metricOutput = [0]
        for i in range(word_probs.shape[0] - 1):
            metricOutput.append(-np.sum(word_probs[i + 1] * np.log(word_probs[i])))
            
    elif metric == "EmbeddingCosines":
        if output.hidden_states is not None:
            embeddings = output.hidden_states[7].detach().numpy()[0]
            metricOutput = [0]
            for i in range(embeddings.shape[0]-1):
                metricOutput.append(cosine(embeddings[i], embeddings[i+1]))
        else:
            raise ValueError("EmbeddingCosines is selected as metric but hidden layers are found in LMOutput.")
            
    else:
        raise ValueError("Invalid metric.")
    
    if transient:
    
        if drift is None:
            raise ValueError("Drift must be provided for transient metric calculations.")
        elif drift < 0.0 or drift > 1.0:
            raise ValueError("Drift must be between 0 and 1.")
            
        else:
            values = metricOutput
            running_avg = [sum(values)/len(values)]
            for i in range(1, len(values)):
                current_avg = running_avg[i-1]
                running_avg.append(current_avg + (values[i] - current_avg) * drift)
            metricOutput = list(map((lambda x: x[0]/x[1] if x[1] > 0 else 0), zip(values, running_avg)))
    
    return metricOutput