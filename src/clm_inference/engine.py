"""Zero-shot CLM decisions using a local MLX encoder and trained heads."""

from clm_inference.schema import (
    Choice, DecisionRequest, DecisionResponse, DecisionUsage, RankRequest,
    RankedCandidate, RankResponse, answer_from_logits, candidates, state_text,
)


class Engine:
    """Score caller-supplied questions and candidates with trained CLM heads."""

    def __init__(self, encoder, heads, model: str, max_tokens: int):
        self.encoder = encoder
        self.heads = heads
        self.model = model
        self.max_tokens = max_tokens

    def answer(self, request: DecisionRequest) -> DecisionResponse:
        """Encode context and options and return the heads' typed answers."""
        pairs = [
            (identifier, question, *candidates(question))
            for identifier, question in request.questions.items()
        ]
        state_vectors, state_tokens = self.encoder.embed(
            [state_text(request.state, question.instructions)
             for _, question, _, _ in pairs], self.max_tokens,
        )
        action_vectors, action_tokens = self.encoder.embed(
            [text for _, _, _, texts in pairs for text in texts],
            self.max_tokens,
        )
        logits = self.heads.logits(state_vectors, action_vectors)
        answers = {}
        offset = 0
        for row, (identifier, question, keys, texts) in enumerate(pairs):
            values = logits[row][offset:offset + len(texts)]
            answers[identifier] = answer_from_logits(
                question, keys,
                [value / request.temperature for value in values],
            )
            offset += len(texts)
        return DecisionResponse(
            model=self.model, answers=answers,
            usage=DecisionUsage(
                billing_units=len(pairs),
                input_tokens=state_tokens + action_tokens,
            ),
        )

    def rank(self, request: RankRequest) -> RankResponse:
        """Rank answers through the same scoring path as a choice question."""
        question = Choice(
            type="choice", instructions=request.question,
            criteria={str(index): answer
                      for index, answer in enumerate(request.answers)},
        )
        result = self.answer(DecisionRequest(
            state=request.context, questions={"rank": question},
            temperature=request.temperature,
        ))
        probabilities = result.answers["rank"].probabilities
        order = sorted(
            probabilities, key=probabilities.__getitem__, reverse=True
        )
        return RankResponse(
            model=self.model,
            ranked=[
                RankedCandidate(
                    rank=rank + 1, candidate=request.answers[int(index)],
                    prob=probabilities[index],
                )
                for rank, index in enumerate(order)
            ],
        )
