from transformers import Trainer
import torch


class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.temperature = config["train"]["temperature"]
        self.ce_loss_fct = torch.nn.KLDivLoss(reduction="batchmean")

    def compute_loss(self, model, inputs):
        student_outputs = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )  # (bs, seq_length, voc_size)
        with torch.no_grad():
            teacher_outputs = self.teacher(
                inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )  # (bs, seq_length, voc_size)

        s_logits, t_logits = student_outputs["logits"], teacher_outputs["logits"]
        mask = inputs["attention_mask"].unsqueeze(-1).expand_as(s_logits)  # (bs, seq_length, voc_size)
        s_logits_slct = torch.masked_select(s_logits, mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        s_logits_slct = s_logits_slct.view(-1, s_logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
        t_logits_slct = torch.masked_select(t_logits, mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        t_logits_slct = t_logits_slct.view(-1, s_logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask  
        loss = (
            self.ce_loss_fct(
                torch.nn.functional.log_softmax(s_logits_slct / self.temperature, dim=-1),
                torch.nn.functional.softmax(t_logits_slct / self.temperature, dim=-1),
            )
            * (self.temperature) ** 2
        )
        return loss
        