package br.cafedl.game.viewmodel;

import br.cafedl.game.model.GameSession;

public abstract class ViewModel {
    protected GameSession session;

    public ViewModel(GameSession session) {
        this.session = session;
    }

    public abstract void updateView();
}
