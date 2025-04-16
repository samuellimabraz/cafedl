package br.cafedl.game.viewmodel;

import br.cafedl.game.model.GameSession;
import br.cafedl.game.controller.MenuController;

public class MenuViewModel extends ViewModel {
    private final MenuController controller;
    public MenuViewModel(MenuController menuController, GameSession session) {
        super(session);
        this.controller = menuController;
    }

    @Override
    public void updateView() {}
}
