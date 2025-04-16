package br.cafedl.game.viewmodel;

import br.cafedl.game.model.GameSession;
import br.cafedl.game.controller.TransitionController;

public class TransitionViewModel extends ViewModel {
    private final TransitionController controller;

    public TransitionViewModel(TransitionController controller, GameSession session) {
        super(session);
        this.controller = controller;
    }

    public void startRound() {
        session.start();
        updateView();
    }

    @Override
    public void updateView() {
        // Update the view based on the data in session
        // For example, you might want to update the current round and category
        controller.setRound(session.getCurrentRoundIndex() + 1);
        controller.setCategory(session.getCurrentRound().getCategory());
    }
}
