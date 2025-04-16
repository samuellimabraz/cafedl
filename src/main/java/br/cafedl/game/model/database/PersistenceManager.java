package br.cafedl.game.model.database;

import br.cafedl.game.model.Draw;
import br.cafedl.game.model.GameSession;
import br.cafedl.game.model.Round;

import javax.persistence.EntityManager;
import javax.persistence.EntityTransaction;
import javax.persistence.Persistence;
import javax.persistence.TypedQuery;
import java.time.LocalTime;
import java.util.Arrays;
import java.util.List;

public class PersistenceManager {

    public static void persist(EntityManager entityManager, Object object) {
        EntityTransaction transaction = entityManager.getTransaction();
        try {
            transaction.begin();
            entityManager.persist(object);
            transaction.commit();
            System.out.println("Persisted object: " + object);
        } finally {
            if (transaction.isActive()) {
                transaction.rollback();
            }
        }
    }

    public static void persistAll(EntityManager entityManager, List<Object> objects) {
        EntityTransaction transaction = entityManager.getTransaction();
        try {
            transaction.begin();
            for (Object object : objects) {
                entityManager.persist(object);
                System.out.println("Persisted object: " + object);
            }
            transaction.commit();
        } finally {
            if (transaction.isActive()) {
                transaction.rollback();
            }
        }
    }

    public static void updateEndTime(EntityManager entityManager, Long sessionId, LocalTime endTime) {
        EntityTransaction transaction = entityManager.getTransaction();
        try {
            transaction.begin();
            GameSession session = entityManager.find(GameSession.class, sessionId);
            if (session != null) {
                session.setEndTime(endTime);
                entityManager.merge(session);
            }
            transaction.commit();
            System.out.println("Updated endTime for session: " + sessionId);
        } finally {
            if (transaction.isActive()) {
                transaction.rollback();
            }
        }
    }

    public static EntityManager createEntityManager(String persistenceUnit) {
        return Persistence.createEntityManagerFactory(persistenceUnit).createEntityManager();
    }

    public static List<String> getAllCategories(EntityManager entityManager) {
        TypedQuery<String> query = entityManager.createQuery(
                "SELECT DISTINCT pr.category FROM PredictionResult pr", String.class);
        return query.getResultList();
    }

    public static List<Draw> getDrawingsByCategory(EntityManager entityManager, String category) {
        TypedQuery<Draw> query = entityManager.createQuery(
                "SELECT d FROM Draw d WHERE d.predictionResult.category = :category", Draw.class);
        query.setParameter("category", category);
        return query.getResultList();
    }

    public static Round getRoundByDrawId(EntityManager entityManager, Long drawId) {
        TypedQuery<Round> query = entityManager.createQuery(
                "SELECT r FROM Round r WHERE r.drawing.id = :drawId", Round.class);
        query.setParameter("drawId", drawId);
        return query.getSingleResult();
    }

    public static void deleteDraw(EntityManager entityManager, Long drawId) {
        EntityTransaction transaction = entityManager.getTransaction();
        try {
            transaction.begin();
            Draw draw = entityManager.find(Draw.class, drawId);
            if (draw != null) {
                // Remove all rounds associated with the draw
                Round round = getRoundByDrawId(entityManager, drawId);
                entityManager.remove(round);
                entityManager.remove(draw.getPredictionResult());
                entityManager.remove(draw);
            }
            transaction.commit();
        } finally {
            if (transaction.isActive()) {
                transaction.rollback();
            }
        }
    }

    public static void main(String[] args) {
        System.out.println("Test");
        EntityManager em = createEntityManager("quickdrawPU");
        List<String> categories = getAllCategories(em);
        System.out.println(categories);
        List<Draw> drawings = getDrawingsByCategory(em, "hourglass");
        System.out.println(drawings.size());
        for (Draw drawing : drawings) {
            System.out.println(drawing.getCategory());
            System.out.println(drawing.getConfidence());
            System.out.println(Arrays.toString(drawing.getData()));
        }
    }
}
