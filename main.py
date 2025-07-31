import json
import numpy as np
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from nltk.translate import meteor_score, nist_score, ribes_score, gleu_score
import sacrebleu
from tqdm import tqdm


class TextEvaluator:
    def __init__(self, all_evaluation_indicators=None):
        self.pred_data = None
        self.truth_data = None
        if all_evaluation_indicators is None:
            all_evaluation_indicators = ['METEOR', 'RIBES', 'GLEU', 'NIST', 'chrF', 'chrF++', 'BLEU-1', 'BLEU-2', 
                                         'BLEU-3', 'BLEU-4', 'CIDEr', 'ROUGE']
        self.all_evaluation_indicators = all_evaluation_indicators
        self.metrics = {}
        self.scorers = {
            'BLEU': Bleu(n=4),
            'CIDEr': Cider(),
            'ROUGE': Rouge()
        }

    def load_data(self, truth_path, pred_path):
        with open(truth_path, 'r', encoding='utf-8') as f:
            self.truth_data = json.load(f)
        with open(pred_path, 'r', encoding='utf-8') as f:
            self.pred_data = json.load(f)

    def calculate_metrics(self, references, hypotheses):
        scores = {}
        hypothesis_scores = []
        for hyp in hypotheses:
            hyp_tokens = hyp.split()
            metric_scores = {
                'METEOR': meteor_score.meteor_score([ref.split() for ref in references], hyp_tokens),
                'RIBES': ribes_score.sentence_ribes([ref.split() for ref in references], hyp_tokens),
                'GLEU': gleu_score.sentence_gleu([ref.split() for ref in references], hyp_tokens),
                'chrF': sacrebleu.sentence_chrf(hypothesis=hyp, references=[ref for ref in references]).score / 100.0,
                'chrF++': sacrebleu.sentence_chrf(hypothesis=hyp, references=[ref for ref in references], word_order=2).score / 100.0,
            }
            try:
                metric_scores['NIST'] = nist_score.sentence_nist([ref.split() for ref in references], hyp_tokens)
            except Exception:
                metric_scores['NIST'] = 0.0

            for scorer_name, scorer in self.scorers.items():
                gts = {'caption': [ref for ref in references]}
                res = {'caption': [hyp]}
                score_value, _ = scorer.compute_score(gts, res)

                if scorer_name == 'BLEU':
                    for j, bleu_score in enumerate(score_value, start=1):
                        metric_scores[f'BLEU-{j}'] = bleu_score
                else:
                    metric_scores[scorer_name] = score_value

            hypothesis_scores.append(metric_scores)

        for metric in self.all_evaluation_indicators:
            scores[metric] = np.max([h_score[metric] for h_score in hypothesis_scores])
        return scores

    def evaluate(self, truth_path, pred_path):
        self.load_data(truth_path, pred_path)
        sentence_scores = {metric: [] for metric in self.all_evaluation_indicators}

        for key, refs in tqdm(self.truth_data.items(), desc='calculating metrics'):
            if key in self.pred_data:
                hyps = self.pred_data[key]
                scores = self.calculate_metrics(refs, hyps)
                for metric, score in scores.items():
                    sentence_scores[metric].append(score)

        self.metrics.update({
            metric: np.mean(scores)
            for metric, scores in sentence_scores.items()
        })

        return self.metrics

    def print_results(self):
        print("\n=== Evaluation ===")
        for metric in sorted(self.metrics.keys()):
            print(f"{metric}: {self.metrics[metric]:.4f}")


if __name__ == "__main__":
    evaluator = TextEvaluator()
    metrics = evaluator.evaluate('examples/original_captions.json', 'examples/generated_captions.json')
    evaluator.print_results()
