package br.cafedl.game.model.database;


import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;

public class GeolocationUtil {
    private static final String IPINFO_URL = "https://ipinfo.io/country";

    public static String getCountryFromIP() {
        String country = "US"; // Default country

        try {
            URL url = new URL(IPINFO_URL);
            HttpURLConnection con = (HttpURLConnection) url.openConnection();
            con.setRequestMethod("GET");
            con.setRequestProperty("User-Agent", "Mozilla/5.0");
            int responseCode = con.getResponseCode();
            if (responseCode == HttpURLConnection.HTTP_OK) {
                BufferedReader in = new BufferedReader(new InputStreamReader(con.getInputStream()));
                String inputLine;
                StringBuffer response = new StringBuffer();
                while ((inputLine = in.readLine()) != null) {
                    response.append(inputLine);
                }
                in.close();
                country = response.toString();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return country;
    }

    public static void main(String[] args) {
        System.out.println("Country: " + getCountryFromIP());
    }
}

