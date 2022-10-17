class OutputDistributionCallback(Callback):
    """
    Plot histogram of predictions for each class. This callback supports binary & multi-classs predictions
    """

    def __init__(
        self,
        targets_key: str,
        output_key: str,
        output_activation: Optional[Callable],
        num_classes: int,
        prefix="distribution",
        ignore_index=None,
    ):
        """

        Args:
            targets_key:
            output_key:
            output_activation: A function that should convert logits to class labels
            For binary predictions this could be `lambda x: int(x > 0.5)` or `lambda x: torch.argmax(x, dim=1)`
            for multi-class predictions.
            num_classes: Number of classes. Must be 2 for binary.
            prefix:
        """
        super().__init__(CallbackOrder.Metric)
        self.prefix = prefix
        self.targets_key = targets_key
        self.output_key = output_key
        self.true_labels = []
        self.pred_labels = []
        self.num_classes = num_classes
        self.output_activation = output_activation
        self.ignore_index = ignore_index

    def on_loader_start(self, state: IRunner):
        self.true_labels = []
        self.pred_labels = []

    @torch.no_grad()
    def on_batch_end(self, state: IRunner):
        y_trues = state.input[self.targets_key].detach()
        y_preds = state.output[self.output_key].detach().float()
        if self.output_activation:
            y_preds = self.output_activation(y_preds)

        y_trues = to_numpy(y_trues).reshape(-1)
        y_preds = to_numpy(y_preds).reshape(-1)

        if self.ignore_index is not None:
            include_mask = y_trues != self.ignore_index
            y_trues = y_trues[include_mask]
            y_preds = y_preds[include_mask]

        self.true_labels.extend(y_trues)
        self.pred_labels.extend(y_preds)

    def on_loader_end(self, state: IRunner):
        true_labels = np.concatenate(all_gather(np.array(self.true_labels)))
        pred_probas = np.concatenate(all_gather(np.array(self.pred_labels)))

        if is_main_process():
            logger = get_tensorboard_logger(state)

            for class_label in range(self.num_classes):
                p = pred_probas[true_labels == class_label]
                if p.any():
                    logger.add_histogram(
                        tag=f"{self.prefix}/{class_label}",
                        values=pred_probas[true_labels == class_label],
                        global_step=state.global_epoch,
                    )
