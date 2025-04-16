package br.cafedl.game.model;


import javax.persistence.*;
import java.util.Timer;
import java.util.TimerTask;

@Entity
public class Round {
    private final String category;
    private int time;

    @Transient
    private boolean isPaused = false;

    @Transient
    private Timer timer;

    @OneToOne(cascade = CascadeType.ALL, fetch = FetchType.EAGER, orphanRemoval = true)
    private Draw drawing;

    @Transient
    private RoundListener listener;

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    public Round() {
        this.category = "";
        this.time = 0;
        this.timer = new Timer();
    }

    public Round(String category, int time) {
        this.category = category;
        this.time = time;
        this.timer = new Timer();
    }

    public void setListener(RoundListener listener) {
        this.listener = listener;
    }

    public String getCategory() {
        return category;
    }

    public int getTime() {
        return time;
    }

    public Timer getTimer() {
        return timer;
    }

    public void setTime(int time) {
        this.time = time;
    }

    public void setDrawing(Draw drawing) {
        this.drawing = drawing;
    }

    public Draw getDrawing() { return this.drawing; }

    public void start() {
        if (timer != null) {
            timer.cancel();
        }
        timer = new Timer();
        timer.scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                time--;
                if (listener != null) {
                    listener.onTimeUpdated(time);
                }
                if (time == 0) {
                    timer.cancel();
                }
            }
        }, 0, 1000);
    }

    public void pause() {
        if (timer != null) {
            timer.cancel();
            timer = null;
            isPaused = true;
        }
    }

    public void resume() {
        if (isPaused) {
            timer = new Timer();
            timer.scheduleAtFixedRate(new TimerTask() {
                @Override
                public void run() {
                    time--;
                    if (listener != null) {
                        listener.onTimeUpdated(time);
                    }
                    if (time == 0) {
                        timer.cancel();
                    }
                }
            }, 0, 1000);
            isPaused = false;
        }
    }

    public void setId(Long id) {
        this.id = id;
    }

    public Long getId() {
        return id;
    }
}