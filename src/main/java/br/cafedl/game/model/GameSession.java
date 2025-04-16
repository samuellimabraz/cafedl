package br.cafedl.game.model;

import br.cafedl.game.model.database.GeolocationUtil;
import br.cafedl.game.viewmodel.GameViewModel;

import javax.persistence.*;
import java.time.LocalDate;
import java.time.LocalTime;
import java.util.ArrayList;
import java.util.List;

@Entity
public class GameSession implements RoundListener {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Transient
    private GameViewModel viewModel;

    @Transient
    private List<String> roundCategories;
    @Transient
    private final int maxRounds = 4;

    @OneToMany(cascade = CascadeType.ALL, fetch = FetchType.LAZY)
    @JoinColumn(name = "game_session_id", nullable = false)
    private List<Round> rounds;

    private int currentRound;

    @OneToOne
    private PredictionModel model;

    private LocalDate date;
    private LocalTime startTime;
    private LocalTime endTime;
    private String country;

    public GameSession(PredictionModel model) {
        this.model = model;
        currentRound = 0;
        roundCategories = new ArrayList<>(maxRounds);
        rounds = new ArrayList<>(maxRounds);

        List<String> categories = new ArrayList<>(model.getCategories());
        for (int i = 0; i < maxRounds; i++) {
            // Add random category of model categories
            String category = categories.get((int) (Math.random() * categories.size()));
            categories.remove(category);
            roundCategories.add(category);
        }

        this.date = LocalDate.now();
        this.startTime = LocalTime.now();
        this.country = GeolocationUtil.getCountryFromIP();

        // Print the country and date information
        System.out.println("Game started in country: " + country + " at " + date + " " + startTime);
    }

    public GameSession() {
    }

    public void initRounds() {
        rounds.add(new Round(roundCategories.get(0), 20));
        rounds.get(0).setListener(this);
    }

    public void start() {
        // Start first round
        rounds.get(currentRound).start();
    }

    public boolean nextRound() {
        cancelTimer();
        currentRound++;
        if (currentRound >= roundCategories.size()) {
            return false;
        }
        rounds.add(new Round(roundCategories.get(currentRound), 20));
        rounds.get(currentRound).setListener(this);
        return true;
    }
    public void endGame() {
        cancelTimer();
        this.endTime = LocalTime.now();
    }

    public void cancelTimer() {
        try {
            getCurrentRound().getTimer().cancel();
        } catch (Exception e) {
            //System.out.println("No timer to cancel");
        }
    }

    public Round getCurrentRound() {
        return rounds.get(currentRound);
    }

    public int getCurrentRoundIndex() {
        return currentRound;
    }

    public PredictionModel getModel() {
        return model;
    }

    public List<Round> getRounds() {
        return rounds;
    }

    public void setDrawing(Draw drawing) {
        this.rounds.get(currentRound).setDrawing(drawing);
    }

    public Draw getDrawing() {
        return rounds.get(currentRound).getDrawing();
    }

    public List<PredictionResult> predict() {
        return model.predict(rounds.get(currentRound).getDrawing().getData());
    }

    public void setViewModel(GameViewModel viewModel) {
        this.viewModel = viewModel;
    }

    public LocalDate getDate() {
        return date;
    }

    public String getCountry() {
        return country;
    }

    public LocalTime getStartTime() {
        return startTime;
    }

    public LocalTime getEndTime() {
        return endTime;
    }

    public void setEndTime(LocalTime endTime) {
        this.endTime = endTime;
    }

    public Long getId() {
        return id;
    }

    @Override
    public void onTimeUpdated(int time) {
        viewModel.updateTimer(time);
    }
}
