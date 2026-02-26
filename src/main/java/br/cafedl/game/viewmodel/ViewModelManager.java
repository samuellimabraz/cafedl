package br.cafedl.game.viewmodel;

import br.cafedl.game.model.GameSession;
import br.cafedl.game.model.PredictionModel;
import br.cafedl.game.controller.GameController;
import br.cafedl.game.controller.TransitionController;
import br.cafedl.game.model.database.PersistenceManager;

import jakarta.persistence.EntityManager;

public class ViewModelManager {
    private static ViewModelManager instance = new ViewModelManager();
    private GameSession currentSession;
    private EntityManager entityManager;

    private ViewModelManager() {}

    public static ViewModelManager getInstance() {
        return instance;
    }

    public void startNewSession(PredictionModel model, EntityManager em) {
        currentSession = new GameSession(model);
        entityManager = em;
        if (em != null) {
            System.out.println("Database Connected created");
            PersistenceManager.persist(em, currentSession.getModel());
            PersistenceManager.persist(em, currentSession);
        }
        currentSession.initRounds();
        System.out.println("New Session started");
        System.out.println("Model: " + currentSession.getModel().getClass().getSimpleName());
    }

    public GameSession getCurrentSession() {
        return currentSession;
    }

    public GameViewModel getGameViewModel(GameController controller) {
        return new GameViewModel(controller, currentSession, entityManager);
    }

    public TransitionViewModel getTransitionViewModel(TransitionController controller) {
        return new TransitionViewModel(controller, currentSession);
    }
}
