import json

# Patterns for design patterns with placeholders
patterns = [
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class PhoneDisplay implements Observer {{\n    @Override\n    public void update(float temperature) {{\n        System.out.println(\"Phone Display: Temperature updated to \" + temperature + \"\u00c2\u00b0C\");\n    }}\n}}\n\nclass TVDisplay implements Observer {{\n    @Override\n    public void update(float temperature) {{\n        System.out.println(\"TV Display: Temperature updated to \" + temperature + \"\u00c2\u00b0C\");\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} station = new {name}();\n        Observer phone = new PhoneDisplay();\n        Observer tv = new TVDisplay();\n        \n        station.addObserver(phone);\n        station.addObserver(tv);\n        station.setTemperature(25);\n        station.setTemperature(30);\n    }}\n}}",
        "complexity": "Intermediate",
        "language": "Java",
        "context": "Weather station notifies all subscribed displays about temperature updates",
        "edge_cases": [
            "Handling observer removal dynamically",
            "Multiple updates in a short time period"
        ],
        "dependencies": [
            "None"
        ],
        "performance_notes": "Efficiently updates all observers when data changes",
        "real_world_usage": "Used in IoT-based weather monitoring systems",
        "testing_notes": "Simulate rapid temperature changes and observer disconnections",
        "comments": "Demonstrates real-time data propagation using Observer Pattern",
        "source": "Inspired by weather monitoring applications"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class CryptoTrader implements Observer {{\n    private String name;\n    \n    public CryptoTrader(String name) {{\n        this.name = name;\n    }}\n    \n    @Override\n    public void update(String crypto, double price) {{\n        System.out.println(name + \" received update: \" + crypto + \" price changed to $\" + price);\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} exchange = new {name}();\n        Observer trader1 = new CryptoTrader(\"Alice\");\n        Observer trader2 = new CryptoTrader(\"Bob\");\n        \n        exchange.addObserver(trader1);\n        exchange.addObserver(trader2);\n        \n        exchange.notifyObservers(\"Bitcoin\", 45000.0);\n        exchange.notifyObservers(\"Ethereum\", 3200.5);\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "Crypto exchange notifies traders about price changes",
        "edge_cases": [
            "Handling rapid price fluctuations",
            "Notifying thousands of traders efficiently"
        ],
        "dependencies": [
            "Real-time market data feed"
        ],
        "performance_notes": "Optimized for handling high-frequency trading updates",
        "real_world_usage": "Used in crypto trading platforms like Binance, Coinbase",
        "testing_notes": "Simulate market volatility and large-scale trading activities",
        "comments": "Supports integration with trading bots for automated responses",
        "source": "Inspired by real-time cryptocurrency trading platforms"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class SportsFan implements Observer {{\n    private String name;\n    \n    public SportsFan(String name) {{\n        this.name = name;\n    }}\n    \n    @Override\n    public void update(String match, String score) {{\n        System.out.println(name + \" received update: \" + match + \" score is \" + score);\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} server = new {name}();\n        Observer fan1 = new SportsFan(\"John\");\n        Observer fan2 = new SportsFan(\"Emma\");\n        \n        server.addObserver(fan1);\n        server.addObserver(fan2);\n        \n        server.notifyObservers(\"Soccer: Team A vs Team B\", \"2-1\");\n        server.notifyObservers(\"Basketball: Lakers vs Celtics\", \"101-99\");\n    }}\n}}",
        "complexity": "Intermediate",
        "language": "Java",
        "context": "Live sports broadcasting service provides real-time score updates",
        "edge_cases": [
            "Handling multiple games simultaneously",
            "Ensuring real-time notifications with low latency"
        ],
        "dependencies": [
            "Live sports data feed APIs"
        ],
        "performance_notes": "Ensures efficient event-driven updates for subscribers",
        "real_world_usage": "Used in ESPN, FIFA, NBA live score updates",
        "testing_notes": "Simulate real-time match score updates",
        "comments": "Scalable for supporting multiple sports and leagues",
        "source": "Inspired by real-time sports streaming applications"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class Driver implements Observer {{\n    private String name;\n    \n    public Driver(String name) {{\n        this.name = name;\n    }}\n    \n    @Override\n    public void update(String location, String status) {{\n        System.out.println(name + \" received traffic update: \" + location + \" is \" + status);\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} system = new {name}();\n        Observer driver1 = new Driver(\"Alice\");\n        Observer driver2 = new Driver(\"Bob\");\n        \n        system.addObserver(driver1);\n        system.addObserver(driver2);\n        \n        system.notifyObservers(\"Highway A1\", \"Congested\");\n        system.notifyObservers(\"Downtown Street\", \"Clear\");\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "Traffic monitoring system notifies users about real-time traffic conditions",
        "edge_cases": [
            "Handling large-scale real-time updates",
            "Notifying thousands of users efficiently"
        ],
        "dependencies": [
            "GPS tracking, real-time traffic APIs"
        ],
        "performance_notes": "Optimized for high-performance, real-time event updates",
        "real_world_usage": "Used in Google Maps, Waze, Apple Maps",
        "testing_notes": "Simulate peak-hour traffic conditions",
        "comments": "Supports integration with smart city infrastructure",
        "source": "Inspired by real-time traffic monitoring applications"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class BrandManager implements Observer {{\n    private String brandName;\n    \n    public BrandManager(String brandName) {{\n        this.brandName = brandName;\n    }}\n    \n    @Override\n    public void update(String brand, String sentiment) {{\n        if (this.brandName.equals(brand)) {{\n            System.out.println(\"ALERT: Sentiment for \" + brand + \" changed to: \" + sentiment);\n        }}\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} system = new {name}();\n        Observer brand1 = new BrandManager(\"Nike\");\n        Observer brand2 = new BrandManager(\"Adidas\");\n        \n        system.addObserver(brand1);\n        system.addObserver(brand2);\n        \n        system.analyzeTweet(\"Nike\", \"Positive\");\n        system.analyzeTweet(\"Adidas\", \"Negative\");\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "AI-based system analyzes sentiment from social media and alerts brand managers",
        "edge_cases": [
            "Detecting sarcasm and ambiguous sentiment",
            "Handling a high volume of real-time social media posts"
        ],
        "dependencies": [
            "Natural Language Processing (NLP) APIs, Twitter API"
        ],
        "performance_notes": "Ensures near-instant sentiment updates for brands",
        "real_world_usage": "Used in AI-powered brand monitoring platforms like Brandwatch",
        "testing_notes": "Simulate positive and negative sentiment spikes for brands",
        "comments": "Can be expanded to support multiple sentiment sources",
        "source": "Inspired by AI-driven social media analysis tools"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class Doctor implements Observer {{\n    private String name;\n    \n    public Doctor(String name) {{\n        this.name = name;\n    }}\n    \n    @Override\n    public void update(String patient, String vitalSign, double value) {{\n        System.out.println(\"ALERT for Dr. \" + name + \": \" + patient + \"'s \" + vitalSign + \" is critical at \" + value);\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} system = new {name}();\n        Observer doctor1 = new Doctor(\"Smith\");\n        Observer doctor2 = new Doctor(\"Jones\");\n        \n        system.addObserver(doctor1);\n        system.addObserver(doctor2);\n        \n        system.monitorPatient(\"John Doe\", \"Heart Rate\", 180);\n        system.monitorPatient(\"Jane Doe\", \"Blood Pressure\", 90);\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "AI-powered patient monitoring system alerts doctors when vitals are abnormal",
        "edge_cases": [
            "Avoiding false alarms for minor fluctuations",
            "Ensuring real-time alerts for critical cases"
        ],
        "dependencies": [
            "IoT medical devices, AI-based anomaly detection"
        ],
        "performance_notes": "Ensures low-latency emergency notifications",
        "real_world_usage": "Used in ICU and emergency monitoring systems",
        "testing_notes": "Simulate various emergency and normal patient vitals",
        "comments": "Can integrate with wearable health monitors for real-time tracking",
        "source": "Inspired by AI-driven healthcare monitoring solutions"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class AutonomousCar implements Observer {{\n    private String carID;\n    \n    public AutonomousCar(String carID) {{\n        this.carID = carID;\n    }}\n    \n    @Override\n    public void update(String vehicle, String status) {{\n        System.out.println(\"WARNING for \" + carID + \": \" + vehicle + \" is \" + status);\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} system = new {name}();\n        Observer car1 = new AutonomousCar(\"Tesla Model X\");\n        Observer car2 = new AutonomousCar(\"Waymo One\");\n        \n        system.addObserver(car1);\n        system.addObserver(car2);\n        \n        system.detectHazard(\"Truck\", \"Sudden Braking\");\n        system.detectHazard(\"Motorcycle\", \"Lane Changing\");\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "Autonomous vehicles use AI-based collision detection and alerting",
        "edge_cases": [
            "Handling sudden unexpected obstacles",
            "Minimizing false collision alerts"
        ],
        "dependencies": [
            "LIDAR, AI object detection models"
        ],
        "performance_notes": "Optimized for real-time hazard detection",
        "real_world_usage": "Used in Tesla, Waymo, and self-driving vehicles",
        "testing_notes": "Simulate various traffic conditions and emergency scenarios",
        "comments": "Can integrate with vehicle-to-vehicle communication systems",
        "source": "Inspired by autonomous driving technology"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class SecurityAnalyst implements Observer {{\n    private String name;\n    \n    public SecurityAnalyst(String name) {{\n        this.name = name;\n    }}\n    \n    @Override\n    public void update(String eventType, String details) {{\n        System.out.println(\"ALERT for \" + name + \": \" + eventType + \" detected - \" + details);\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} ids = new {name}();\n        Observer analyst1 = new SecurityAnalyst(\"Alice\");\n        Observer analyst2 = new SecurityAnalyst(\"Bob\");\n        \n        ids.addObserver(analyst1);\n        ids.addObserver(analyst2);\n        \n        ids.detectIntrusion(\"Unauthorized Access\", \"IP: 192.168.1.100\");\n        ids.detectIntrusion(\"Malware Attack\", \"Trojan detected in system logs\");\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "AI-based system detects cyber threats and alerts security analysts",
        "edge_cases": [
            "Reducing false positives",
            "Handling large-scale network traffic efficiently"
        ],
        "dependencies": [
            "AI anomaly detection, network monitoring APIs"
        ],
        "performance_notes": "Optimized for real-time cybersecurity threat detection",
        "real_world_usage": "Used in firewalls, intrusion detection systems (IDS), and SOCs",
        "testing_notes": "Simulate various types of cyberattacks",
        "comments": "Can be integrated with machine learning models for predictive threat detection",
        "source": "Inspired by AI-driven cybersecurity platforms"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class SmartLight implements Observer {{\n    @Override\n    public void update(String setting, String value) {{\n        System.out.println(\"Smart Light updated: \" + setting + \" changed to \" + value);\n    }}\n}}\n\nclass SmartThermostat implements Observer {{\n    @Override\n    public void update(String setting, String value) {{\n        System.out.println(\"Smart Thermostat updated: \" + setting + \" changed to \" + value);\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} homeController = new {name}();\n        Observer light = new SmartLight();\n        Observer thermostat = new SmartThermostat();\n        \n        homeController.addObserver(light);\n        homeController.addObserver(thermostat);\n        \n        homeController.changeSetting(\"Brightness\", \"75%\");\n        homeController.changeSetting(\"Temperature\", \"22\u00c2\u00b0C\");\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "Smart home devices adjust settings dynamically based on user preferences",
        "edge_cases": [
            "Ensuring real-time response from all devices",
            "Handling network latency issues"
        ],
        "dependencies": [
            "IoT device APIs, home automation controllers"
        ],
        "performance_notes": "Optimized for low-latency event-driven automation",
        "real_world_usage": "Used in Google Nest, Amazon Alexa, Samsung SmartThings",
        "testing_notes": "Simulate user preference changes and network interruptions",
        "comments": "Can be extended to support security systems and smart locks",
        "source": "Inspired by AI-driven smart home automation"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class TrafficLight implements Observer {{\n    private String location;\n    \n    public TrafficLight(String location) {{\n        this.location = location;\n    }}\n    \n    @Override\n    public void update(String intersection, String trafficStatus) {{\n        if (this.location.equals(intersection)) {{\n            System.out.println(\"Traffic Light at \" + intersection + \" adjusted for \" + trafficStatus);\n        }}\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} controlCenter = new {name}();\n        Observer light1 = new TrafficLight(\"5th Avenue\");\n        Observer light2 = new TrafficLight(\"Main Street\");\n        \n        controlCenter.addObserver(light1);\n        controlCenter.addObserver(light2);\n        \n        controlCenter.updateTraffic(\"5th Avenue\", \"Heavy Traffic\");\n        controlCenter.updateTraffic(\"Main Street\", \"Light Traffic\");\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "AI-powered traffic lights adjust based on congestion levels",
        "edge_cases": [
            "Handling sudden emergency vehicles",
            "Ensuring low-latency responses"
        ],
        "dependencies": [
            "AI-based traffic monitoring systems, IoT sensors"
        ],
        "performance_notes": "Optimized for real-time congestion control",
        "real_world_usage": "Used in smart city infrastructure like Google Traffic",
        "testing_notes": "Simulate traffic surges and sensor failures",
        "comments": "Can be extended for vehicle-to-vehicle (V2V) communication",
        "source": "Inspired by AI-driven smart city initiatives"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class EmergencyResponseTeam implements Observer {{\n    private String teamName;\n    \n    public EmergencyResponseTeam(String teamName) {{\n        this.teamName = teamName;\n    }}\n    \n    @Override\n    public void update(String location, double magnitude) {{\n        System.out.println(teamName + \" ALERT: Earthquake detected at \" + location + \" with magnitude \" + magnitude);\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} system = new {name}();\n        Observer team1 = new EmergencyResponseTeam(\"Red Cross\");\n        Observer team2 = new EmergencyResponseTeam(\"FEMA\");\n        \n        system.addObserver(team1);\n        system.addObserver(team2);\n        \n        system.detectEarthquake(\"California\", 6.5);\n        system.detectEarthquake(\"Tokyo\", 7.2);\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "AI-powered seismic monitoring system detects earthquakes and alerts emergency teams",
        "edge_cases": [
            "Minimizing false alarms",
            "Handling earthquakes in remote locations with no sensors"
        ],
        "dependencies": [
            "Seismic data APIs, AI-based anomaly detection models"
        ],
        "performance_notes": "Optimized for ultra-low-latency alerts",
        "real_world_usage": "Used in global earthquake detection and warning systems",
        "testing_notes": "Simulate different magnitudes and earthquake intensities",
        "comments": "Can be extended to support tsunami detection",
        "source": "Inspired by AI-driven disaster response systems"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class Investor implements Observer {{\n    private String name;\n    \n    public Investor(String name) {{\n        this.name = name;\n    }}\n    \n    @Override\n    public void update(String stock, double price, String recommendation) {{\n        System.out.println(name + \" ALERT: \" + stock + \" is at $\" + price + \" - Recommendation: \" + recommendation);\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} system = new {name}();\n        Observer investor1 = new Investor(\"Alice\");\n        Observer investor2 = new Investor(\"Bob\");\n        \n        system.addObserver(investor1);\n        system.addObserver(investor2);\n        \n        system.analyzeStock(\"AAPL\", 150.0, \"BUY\");\n        system.analyzeStock(\"TSLA\", 900.5, \"SELL\");\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "AI-powered stock trading system provides real-time investment alerts",
        "edge_cases": [
            "Handling volatile market changes",
            "Filtering out noise from unreliable trading signals"
        ],
        "dependencies": [
            "Real-time stock market APIs, AI trading models"
        ],
        "performance_notes": "Optimized for low-latency financial trading",
        "real_world_usage": "Used in AI-powered trading platforms like Robinhood, E-Trade",
        "testing_notes": "Simulate different stock market conditions",
        "comments": "Supports algorithmic trading bots and high-frequency trading",
        "source": "Inspired by AI-driven financial trading systems"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class PowerPlant implements Observer {{\n    private String name;\n    \n    public PowerPlant(String name) {{\n        this.name = name;\n    }}\n    \n    @Override\n    public void update(String area, double demand) {{\n        System.out.println(\"Power Plant \" + name + \" adjusting supply for \" + area + \" - Demand: \" + demand + \"MW\");\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} grid = new {name}();\n        Observer plant1 = new PowerPlant(\"Plant A\");\n        Observer plant2 = new PowerPlant(\"Plant B\");\n        \n        grid.addObserver(plant1);\n        grid.addObserver(plant2);\n        \n        grid.reportEnergyUsage(\"New York\", 2500);\n        grid.reportEnergyUsage(\"Los Angeles\", 1800);\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "Smart grid system dynamically adjusts energy distribution based on demand",
        "edge_cases": [
            "Handling sudden spikes in energy demand",
            "Preventing overloading of power plants"
        ],
        "dependencies": [
            "IoT energy meters, AI-based demand forecasting"
        ],
        "performance_notes": "Optimized for real-time energy management",
        "real_world_usage": "Used in modern smart grid infrastructure",
        "testing_notes": "Simulate different energy consumption patterns",
        "comments": "Supports integration with renewable energy sources",
        "source": "Inspired by AI-driven smart grid management systems"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class Pilot implements Observer {{\n    private String flightNumber;\n    \n    public Pilot(String flightNumber) {{\n        this.flightNumber = flightNumber;\n    }}\n    \n    @Override\n    public void update(String flight, String status) {{\n        if (this.flightNumber.equals(flight)) {{\n            System.out.println(\"ALERT for \" + flight + \": \" + status);\n        }}\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} atc = new {name}();\n        Observer flight1 = new Pilot(\"AA101\");\n        Observer flight2 = new Pilot(\"BA202\");\n        \n        atc.addObserver(flight1);\n        atc.addObserver(flight2);\n        \n        atc.detectConflict(\"AA101\", \"Potential collision detected at 35,000 feet\");\n        atc.detectConflict(\"BA202\", \"Severe turbulence ahead\");\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "Air traffic control system monitors flights and alerts pilots about flight hazards",
        "edge_cases": [
            "Minimizing false alarms",
            "Handling multiple flight alerts simultaneously"
        ],
        "dependencies": [
            "Radar tracking, AI-based conflict detection"
        ],
        "performance_notes": "Optimized for real-time communication with pilots",
        "real_world_usage": "Used in aviation traffic management systems",
        "testing_notes": "Simulate different flight conditions and emergencies",
        "comments": "Can integrate with automated autopilot responses",
        "source": "Inspired by AI-driven air traffic control systems"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class Farmer implements Observer {{\n    private String name;\n    \n    public Farmer(String name) {{\n        this.name = name;\n    }}\n    \n    @Override\n    public void update(String field, String status) {{\n        System.out.println(\"ALERT for \" + name + \": \" + field + \" needs attention - \" + status);\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} system = new {name}();\n        Observer farmer1 = new Farmer(\"John\");\n        Observer farmer2 = new Farmer(\"Emily\");\n        \n        system.addObserver(farmer1);\n        system.addObserver(farmer2);\n        \n        system.analyzeSoil(\"North Field\", \"Low moisture level\");\n        system.analyzeSoil(\"South Field\", \"Nutrient deficiency detected\");\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "AI-based system monitors soil conditions and alerts farmers",
        "edge_cases": [
            "Handling seasonal weather variations",
            "Avoiding over-irrigation and wasteful resource usage"
        ],
        "dependencies": [
            "IoT soil sensors, AI crop analysis models"
        ],
        "performance_notes": "Optimized for real-time farm monitoring",
        "real_world_usage": "Used in precision agriculture and automated irrigation systems",
        "testing_notes": "Simulate different soil conditions and crop types",
        "comments": "Can integrate with AI-powered crop health analysis",
        "source": "Inspired by AI-driven smart farming technology"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class MaintenanceEngineer implements Observer {{\n    private String name;\n    \n    public MaintenanceEngineer(String name) {{\n        this.name = name;\n    }}\n    \n    @Override\n    public void update(String machine, String status) {{\n        System.out.println(\"ALERT for \" + name + \": \" + machine + \" - \" + status);\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} system = new {name}();\n        Observer engineer1 = new MaintenanceEngineer(\"Alex\");\n        Observer engineer2 = new MaintenanceEngineer(\"Lisa\");\n        \n        system.addObserver(engineer1);\n        system.addObserver(engineer2);\n        \n        system.detectMachineIssue(\"CNC Machine\", \"Overheating detected\");\n        system.detectMachineIssue(\"Assembly Robot\", \"Sensor malfunction\");\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "AI-driven monitoring system detects industrial equipment failures",
        "edge_cases": [
            "Handling minor fluctuations vs. critical failures",
            "Reducing false alarms to avoid unnecessary downtime"
        ],
        "dependencies": [
            "IoT factory sensors, AI predictive maintenance models"
        ],
        "performance_notes": "Optimized for predictive maintenance with minimal downtime",
        "real_world_usage": "Used in smart manufacturing and Industry 4.0 applications",
        "testing_notes": "Simulate different machine failure scenarios",
        "comments": "Can be extended with AI-driven predictive analytics",
        "source": "Inspired by AI-based industrial maintenance systems"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class CityAuthority implements Observer {{\n    private String department;\n    \n    public CityAuthority(String department) {{\n        this.department = department;\n    }}\n    \n    @Override\n    public void update(String location, double airQualityIndex) {{\n        System.out.println(\"ALERT for \" + department + \": \" + location + \" air quality is critical at \" + airQualityIndex);\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} system = new {name}();\n        Observer authority1 = new CityAuthority(\"Environmental Department\");\n        Observer authority2 = new CityAuthority(\"Public Health Department\");\n        \n        system.addObserver(authority1);\n        system.addObserver(authority2);\n        \n        system.reportAirQuality(\"Downtown\", 180);\n        system.reportAirQuality(\"Industrial Zone\", 250);\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "AI-based pollution monitoring system detects and reports air quality changes",
        "edge_cases": [
            "Handling sensor failures",
            "Ensuring real-time updates for large-scale cities"
        ],
        "dependencies": [
            "IoT air quality sensors, AI-based pollution forecasting models"
        ],
        "performance_notes": "Optimized for real-time air quality monitoring",
        "real_world_usage": "Used in smart cities and environmental protection agencies",
        "testing_notes": "Simulate different pollution scenarios",
        "comments": "Can integrate with traffic control systems to reduce emissions",
        "source": "Inspired by AI-driven environmental monitoring platforms"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class Tutor implements Observer {{\n    private String name;\n    \n    public Tutor(String name) {{\n        this.name = name;\n    }}\n    \n    @Override\n    public void update(String student, String subject, String progress) {{\n        System.out.println(\"Tutor \" + name + \" received update: \" + student + \"'s progress in \" + subject + \" is \" + progress);\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} platform = new {name}();\n        Observer tutor1 = new Tutor(\"Mr. Smith\");\n        Observer tutor2 = new Tutor(\"Mrs. Johnson\");\n        \n        platform.addObserver(tutor1);\n        platform.addObserver(tutor2);\n        \n        platform.updateProgress(\"Alice\", \"Math\", \"Excellent\");\n        platform.updateProgress(\"Bob\", \"Science\", \"Needs Improvement\");\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "AI-driven learning platform dynamically tracks student progress and notifies tutors",
        "edge_cases": [
            "Handling multiple subjects per student",
            "Ensuring unbiased AI assessment of student performance"
        ],
        "dependencies": [
            "AI-based learning models, student tracking databases"
        ],
        "performance_notes": "Optimized for real-time learning adaptation",
        "real_world_usage": "Used in adaptive learning platforms like Coursera, Khan Academy",
        "testing_notes": "Simulate different learning progressions and tutor interventions",
        "comments": "Can be extended with AI-based personalized tutoring recommendations",
        "source": "Inspired by AI-powered education platforms"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class WarehouseManager implements Observer {{\n    private String name;\n    \n    public WarehouseManager(String name) {{\n        this.name = name;\n    }}\n    \n    @Override\n    public void update(String item, int quantity) {{\n        System.out.println(\"Manager \" + name + \" notified: \" + item + \" stock level is \" + quantity);\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} warehouse = new {name}();\n        Observer manager1 = new WarehouseManager(\"Logistics\");\n        Observer manager2 = new WarehouseManager(\"Procurement\");\n        \n        warehouse.addObserver(manager1);\n        warehouse.addObserver(manager2);\n        \n        warehouse.updateStock(\"Laptops\", 50);\n        warehouse.updateStock(\"Mobile Phones\", 20);\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "AI-powered warehouse system tracks and alerts inventory managers about stock levels",
        "edge_cases": [
            "Handling large-scale inventory changes",
            "Minimizing false alerts due to system errors"
        ],
        "dependencies": [
            "IoT warehouse tracking sensors, AI-based demand prediction"
        ],
        "performance_notes": "Optimized for high-speed stock monitoring",
        "real_world_usage": "Used in logistics and e-commerce fulfillment centers",
        "testing_notes": "Simulate inventory fluctuations and warehouse operations",
        "comments": "Can integrate with robotic automation for warehouse management",
        "source": "Inspired by AI-driven supply chain management"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class SmartTrafficLight implements Observer {{\n    private String intersection;\n    \n    public SmartTrafficLight(String intersection) {{\n        this.intersection = intersection;\n    }}\n    \n    @Override\n    public void update(String location, String trafficStatus) {{\n        if (this.intersection.equals(location)) {{\n            System.out.println(\"Traffic light at \" + location + \" adjusting for \" + trafficStatus);\n        }}\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} system = new {name}();\n        Observer light1 = new SmartTrafficLight(\"5th Avenue\");\n        Observer light2 = new SmartTrafficLight(\"Main Street\");\n        \n        system.addObserver(light1);\n        system.addObserver(light2);\n        \n        system.updateTraffic(\"5th Avenue\", \"Heavy Traffic\");\n        system.updateTraffic(\"Main Street\", \"Light Traffic\");\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "AI-driven traffic management system adjusts signals dynamically",
        "edge_cases": [
            "Handling sudden traffic surges",
            "Minimizing unnecessary signal changes"
        ],
        "dependencies": [
            "IoT traffic sensors, AI-based traffic forecasting"
        ],
        "performance_notes": "Optimized for real-time congestion control",
        "real_world_usage": "Used in smart city projects like Google Traffic, Waze",
        "testing_notes": "Simulate various traffic conditions and signal optimizations",
        "comments": "Can integrate with autonomous vehicle traffic control",
        "source": "Inspired by AI-powered urban planning"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class UserNotification implements Observer {{\n    private String userName;\n    \n    public UserNotification(String userName) {{\n        this.userName = userName;\n    }}\n    \n    @Override\n    public void update(String category, double amount) {{\n        System.out.println(\"ALERT for \" + userName + \": You spent $\" + amount + \" on \" + category + \"!\");\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} tracker = new {name}();\n        Observer user1 = new UserNotification(\"Alice\");\n        Observer user2 = new UserNotification(\"Bob\");\n        \n        tracker.addObserver(user1);\n        tracker.addObserver(user2);\n        \n        tracker.trackSpending(\"Restaurants\", 120.50);\n        tracker.trackSpending(\"Shopping\", 300.75);\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "AI-powered budgeting system tracks user spending in real time",
        "edge_cases": [
            "Handling multiple transactions simultaneously",
            "Detecting fraudulent spending behavior"
        ],
        "dependencies": [
            "Banking transaction APIs, AI-based spending pattern analysis"
        ],
        "performance_notes": "Optimized for real-time transaction tracking",
        "real_world_usage": "Used in finance management apps like Mint, YNAB",
        "testing_notes": "Simulate different spending patterns and budget thresholds",
        "comments": "Can be extended with AI-based financial advice",
        "source": "Inspired by AI-driven personal finance management systems"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class EmergencyResponseUnit implements Observer {{\n    private String unitName;\n    \n    public EmergencyResponseUnit(String unitName) {{\n        this.unitName = unitName;\n    }}\n    \n    @Override\n    public void update(String disasterType, String location) {{\n        System.out.println(unitName + \" ALERT: \" + disasterType + \" reported in \" + location + \". Responding immediately!\");\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} system = new {name}();\n        Observer unit1 = new EmergencyResponseUnit(\"Fire Department\");\n        Observer unit2 = new EmergencyResponseUnit(\"Rescue Team\");\n        \n        system.addObserver(unit1);\n        system.addObserver(unit2);\n        \n        system.reportDisaster(\"Earthquake\", \"Los Angeles\");\n        system.reportDisaster(\"Flood\", \"New Orleans\");\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "AI-powered disaster alert system warns response teams in real time",
        "edge_cases": [
            "Minimizing false alarms",
            "Coordinating multiple response teams"
        ],
        "dependencies": [
            "Seismic sensors, AI-based disaster prediction models"
        ],
        "performance_notes": "Optimized for ultra-fast response times",
        "real_world_usage": "Used in disaster management agencies and government response units",
        "testing_notes": "Simulate different natural disaster scenarios",
        "comments": "Can integrate with satellite imaging for better disaster detection",
        "source": "Inspired by AI-driven emergency management systems"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class HealthOrganization implements Observer {{\n    private String name;\n    \n    public HealthOrganization(String name) {{\n        this.name = name;\n    }}\n    \n    @Override\n    public void update(String disease, String location, int cases) {{\n        System.out.println(\"ALERT for \" + name + \": \" + disease + \" outbreak in \" + location + \" with \" + cases + \" cases.\");\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} system = new {name}();\n        Observer who = new HealthOrganization(\"WHO\");\n        Observer cdc = new HealthOrganization(\"CDC\");\n        \n        system.addObserver(who);\n        system.addObserver(cdc);\n        \n        system.reportOutbreak(\"Influenza\", \"New York\", 500);\n        system.reportOutbreak(\"COVID-19\", \"Tokyo\", 2000);\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "AI-driven disease outbreak detection system warns global health organizations",
        "edge_cases": [
            "Minimizing false outbreak reports",
            "Ensuring real-time data accuracy"
        ],
        "dependencies": [
            "AI-based disease prediction models, real-time health data APIs"
        ],
        "performance_notes": "Optimized for global disease tracking",
        "real_world_usage": "Used in WHO, CDC, global pandemic response units",
        "testing_notes": "Simulate different outbreak scenarios",
        "comments": "Can integrate with genetic sequencing for early mutation detection",
        "source": "Inspired by AI-driven epidemiology systems"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class PowerPlant implements Observer {{\n    private String name;\n    \n    public PowerPlant(String name) {{\n        this.name = name;\n    }}\n    \n    @Override\n    public void update(String region, double demand) {{\n        System.out.println(\"Power Plant \" + name + \" adjusting supply for \" + region + \" - Demand: \" + demand + \"MW\");\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} grid = new {name}();\n        Observer plant1 = new PowerPlant(\"Nuclear Plant A\");\n        Observer plant2 = new PowerPlant(\"Hydro Plant B\");\n        \n        grid.addObserver(plant1);\n        grid.addObserver(plant2);\n        \n        grid.reportEnergyDemand(\"California\", 5000);\n        grid.reportEnergyDemand(\"Texas\", 7000);\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "Smart grid system predicts energy demand and adjusts supply dynamically",
        "edge_cases": [
            "Handling sudden energy spikes",
            "Preventing blackouts in high-demand scenarios"
        ],
        "dependencies": [
            "IoT smart meters, AI-based energy forecasting models"
        ],
        "performance_notes": "Optimized for real-time energy management",
        "real_world_usage": "Used in smart grid systems worldwide",
        "testing_notes": "Simulate peak and low energy demand times",
        "comments": "Can integrate with renewable energy sources like solar and wind",
        "source": "Inspired by AI-driven electricity demand management"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class SecurityTeam implements Observer {{\n    private String teamName;\n    \n    public SecurityTeam(String teamName) {{\n        this.teamName = teamName;\n    }}\n    \n    @Override\n    public void update(String transactionID, String alertType) {{\n        System.out.println(\"ALERT for \" + teamName + \": Fraud detected in transaction \" + transactionID + \" - \" + alertType);\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} system = new {name}();\n        Observer team1 = new SecurityTeam(\"Bank Security\");\n        Observer team2 = new SecurityTeam(\"Government Anti-Fraud Unit\");\n        \n        system.addObserver(team1);\n        system.addObserver(team2);\n        \n        system.detectFraud(\"TXN12345\", \"Unusual IP Location\");\n        system.detectFraud(\"TXN67890\", \"Multiple failed login attempts\");\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "AI-powered fraud detection system monitors transactions in real-time",
        "edge_cases": [
            "Minimizing false positives",
            "Detecting complex fraud patterns"
        ],
        "dependencies": [
            "Machine learning anomaly detection models, banking APIs"
        ],
        "performance_notes": "Optimized for ultra-fast transaction monitoring",
        "real_world_usage": "Used in banking, fintech, and payment processing security",
        "testing_notes": "Simulate different types of fraudulent activities",
        "comments": "Can integrate with customer notification systems for fraud alerts",
        "source": "Inspired by AI-driven banking fraud prevention"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class WarehouseManager implements Observer {{\n    private String name;\n    \n    public WarehouseManager(String name) {{\n        this.name = name;\n    }}\n    \n    @Override\n    public void update(String shipmentID, String status) {{\n        System.out.println(\"ALERT for \" + name + \": Shipment \" + shipmentID + \" is \" + status);\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} system = new {name}();\n        Observer manager1 = new WarehouseManager(\"Logistics Hub A\");\n        Observer manager2 = new WarehouseManager(\"Retail Distribution Center\");\n        \n        system.addObserver(manager1);\n        system.addObserver(manager2);\n        \n        system.updateShipment(\"ORD12345\", \"Delayed due to weather\");\n        system.updateShipment(\"ORD67890\", \"Arrived at destination\");\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "AI-powered logistics management system monitors shipment status",
        "edge_cases": [
            "Handling large-scale global shipments",
            "Minimizing false delay notifications"
        ],
        "dependencies": [
            "IoT shipment trackers, AI-based route optimization"
        ],
        "performance_notes": "Optimized for real-time supply chain tracking",
        "real_world_usage": "Used in global logistics companies like FedEx, DHL, Amazon",
        "testing_notes": "Simulate different shipment delays and delivery statuses",
        "comments": "Can integrate with automated warehouse systems",
        "source": "Inspired by AI-driven logistics and supply chain management"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class Farmer implements Observer {{\n    private String name;\n    \n    public Farmer(String name) {{\n        this.name = name;\n    }}\n    \n    @Override\n    public void update(String region, String weatherAlert) {{\n        System.out.println(\"ALERT for \" + name + \": \" + region + \" weather alert - \" + weatherAlert);\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} system = new {name}();\n        Observer farmer1 = new Farmer(\"John\");\n        Observer farmer2 = new Farmer(\"Emily\");\n        \n        system.addObserver(farmer1);\n        system.addObserver(farmer2);\n        \n        system.sendWeatherAlert(\"Texas\", \"Severe Storm Warning\");\n        system.sendWeatherAlert(\"Iowa\", \"Heatwave Advisory\");\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "AI-driven weather alert system monitors and notifies farmers",
        "edge_cases": [
            "Minimizing false weather alerts",
            "Ensuring real-time updates in remote areas"
        ],
        "dependencies": [
            "AI weather prediction models, satellite weather APIs"
        ],
        "performance_notes": "Optimized for real-time meteorological updates",
        "real_world_usage": "Used in precision agriculture, smart farming",
        "testing_notes": "Simulate different weather conditions and alerts",
        "comments": "Can integrate with irrigation control for automated crop protection",
        "source": "Inspired by AI-driven agricultural technology"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class CryptoTrader implements Observer {{\n    private String name;\n    \n    public CryptoTrader(String name) {{\n        this.name = name;\n    }}\n    \n    @Override\n    public void update(String coin, double price) {{\n        System.out.println(\"ALERT for \" + name + \": \" + coin + \" is now $\" + price);\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} exchange = new {name}();\n        Observer trader1 = new CryptoTrader(\"Alice\");\n        Observer trader2 = new CryptoTrader(\"Bob\");\n        \n        exchange.addObserver(trader1);\n        exchange.addObserver(trader2);\n        \n        exchange.notifyPriceChange(\"Bitcoin\", 45000.0);\n        exchange.notifyPriceChange(\"Ethereum\", 3200.5);\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "AI-powered crypto trading system alerts investors about price changes",
        "edge_cases": [
            "Handling high-frequency trading scenarios",
            "Filtering out short-term price fluctuations"
        ],
        "dependencies": [
            "Real-time market APIs, AI price trend analysis"
        ],
        "performance_notes": "Optimized for ultra-fast trading alerts",
        "real_world_usage": "Used in cryptocurrency trading platforms like Binance, Coinbase",
        "testing_notes": "Simulate different price fluctuations and alerts",
        "comments": "Can integrate with AI-driven trading bots for automated investments",
        "source": "Inspired by AI-driven crypto market analysis"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class SupportTeam implements Observer {{\n    private String department;\n    \n    public SupportTeam(String department) {{\n        this.department = department;\n    }}\n    \n    @Override\n    public void update(String ticketID, String issueType) {{\n        System.out.println(\"ALERT for \" + department + \": New ticket \" + ticketID + \" - Issue: \" + issueType);\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} system = new {name}();\n        Observer techSupport = new SupportTeam(\"Technical Support\");\n        Observer billingSupport = new SupportTeam(\"Billing Department\");\n        \n        system.addObserver(techSupport);\n        system.addObserver(billingSupport);\n        \n        system.raiseTicket(\"TCK1001\", \"Login Issue\");\n        system.raiseTicket(\"TCK1002\", \"Billing Discrepancy\");\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "AI-driven ticketing system notifies relevant teams about new customer support issues",
        "edge_cases": [
            "Handling duplicate tickets efficiently",
            "Ensuring priority escalation for urgent issues"
        ],
        "dependencies": [
            "AI-based chatbot integration, CRM ticketing system"
        ],
        "performance_notes": "Optimized for fast response times and auto-routing",
        "real_world_usage": "Used in IT support, customer service platforms like Zendesk, Freshdesk",
        "testing_notes": "Simulate high-ticket volumes and response prioritization",
        "comments": "Can integrate with AI-based ticket classification for automated assignment",
        "source": "Inspired by AI-powered help desk automation"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class PersonalTrainer implements Observer {{\n    private String trainerName;\n    \n    public PersonalTrainer(String trainerName) {{\n        this.trainerName = trainerName;\n    }}\n    \n    @Override\n    public void update(String user, String activity, int caloriesBurned) {{\n        System.out.println(\"ALERT for \" + trainerName + \": \" + user + \" completed \" + activity + \" burning \" + caloriesBurned + \" calories\");\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} tracker = new {name}();\n        Observer trainer1 = new PersonalTrainer(\"Coach Mike\");\n        Observer trainer2 = new PersonalTrainer(\"Coach Lisa\");\n        \n        tracker.addObserver(trainer1);\n        tracker.addObserver(trainer2);\n        \n        tracker.logActivity(\"John\", \"Running\", 500);\n        tracker.logActivity(\"Sarah\", \"Cycling\", 700);\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "AI-powered fitness tracking system notifies trainers about user workouts",
        "edge_cases": [
            "Handling inaccurate sensor data",
            "Minimizing redundant notifications for frequent activities"
        ],
        "dependencies": [
            "Wearable fitness trackers, AI-based health analytics"
        ],
        "performance_notes": "Optimized for real-time fitness coaching and analysis",
        "real_world_usage": "Used in health apps like Apple Health, Google Fit, Fitbit",
        "testing_notes": "Simulate different workout intensities and calorie calculations",
        "comments": "Can integrate with AI-driven nutrition plans based on fitness data",
        "source": "Inspired by AI-powered digital health coaching"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class Homeowner implements Observer {{\n    private String name;\n    \n    public Homeowner(String name) {{\n        this.name = name;\n    }}\n    \n    @Override\n    public void update(String location, String alertType) {{\n        System.out.println(\"ALERT for \" + name + \": Intrusion detected at \" + location + \" - \" + alertType);\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} system = new {name}();\n        Observer owner1 = new Homeowner(\"Alice\");\n        Observer owner2 = new Homeowner(\"Bob\");\n        \n        system.addObserver(owner1);\n        system.addObserver(owner2);\n        \n        system.detectIntrusion(\"Front Door\", \"Unauthorized Access\");\n        system.detectIntrusion(\"Backyard\", \"Motion Detected\");\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "AI-driven home security system detects and notifies about intrusions",
        "edge_cases": [
            "Handling false alarms from pets",
            "Ensuring real-time response in case of actual break-ins"
        ],
        "dependencies": [
            "Smart security cameras, AI facial recognition"
        ],
        "performance_notes": "Optimized for fast and accurate intruder detection",
        "real_world_usage": "Used in smart home security like Ring, Nest Secure",
        "testing_notes": "Simulate different security breach scenarios",
        "comments": "Can integrate with police dispatch for automated emergency response",
        "source": "Inspired by AI-driven home security systems"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class Investor implements Observer {{\n    private String name;\n    \n    public Investor(String name) {{\n        this.name = name;\n    }}\n    \n    @Override\n    public void update(String stock, double price, String trend) {{\n        System.out.println(\"ALERT for \" + name + \": \" + stock + \" is now $\" + price + \" - Trend: \" + trend);\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} system = new {name}();\n        Observer investor1 = new Investor(\"Alice\");\n        Observer investor2 = new Investor(\"Bob\");\n        \n        system.addObserver(investor1);\n        system.addObserver(investor2);\n        \n        system.notifyStockChange(\"Tesla\", 900.5, \"Bullish\");\n        system.notifyStockChange(\"Apple\", 145.0, \"Bearish\");\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "AI-driven stock portfolio tracker alerts investors on market movements",
        "edge_cases": [
            "Handling volatile market fluctuations",
            "Minimizing unnecessary notifications for minor price changes"
        ],
        "dependencies": [
            "Real-time stock market APIs, AI-based stock trend analysis"
        ],
        "performance_notes": "Optimized for high-speed financial data processing",
        "real_world_usage": "Used in financial trading platforms like E-Trade, Robinhood",
        "testing_notes": "Simulate different stock price trends and trading volumes",
        "comments": "Can integrate with AI-powered investment advisors",
        "source": "Inspired by AI-driven stock market monitoring systems"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class LogisticsManager implements Observer {{\n    private String name;\n    \n    public LogisticsManager(String name) {{\n        this.name = name;\n    }}\n    \n    @Override\n    public void update(String vehicleID, String status) {{\n        System.out.println(\"ALERT for \" + name + \": Vehicle \" + vehicleID + \" status: \" + status);\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} system = new {name}();\n        Observer manager1 = new LogisticsManager(\"Warehouse A\");\n        Observer manager2 = new LogisticsManager(\"Retail Hub B\");\n        \n        system.addObserver(manager1);\n        system.addObserver(manager2);\n        \n        system.updateVehicleStatus(\"TRK1023\", \"Delayed due to traffic\");\n        system.updateVehicleStatus(\"TRK2045\", \"Arrived at destination\");\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "AI-powered fleet tracking system provides real-time delivery updates",
        "edge_cases": [
            "Handling unexpected route changes",
            "Optimizing fleet coordination for large-scale deliveries"
        ],
        "dependencies": [
            "GPS tracking APIs, AI-based route optimization models"
        ],
        "performance_notes": "Optimized for real-time vehicle tracking and logistics planning",
        "real_world_usage": "Used in logistics companies like UPS, FedEx, Amazon",
        "testing_notes": "Simulate different delivery route changes and delays",
        "comments": "Can integrate with AI-powered predictive delivery time estimation",
        "source": "Inspired by AI-driven supply chain management"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class Supplier implements Observer {{\n    private String name;\n    \n    public Supplier(String name) {{\n        this.name = name;\n    }}\n    \n    @Override\n    public void update(String product, int stockLevel) {{\n        System.out.println(\"ALERT for \" + name + \": \" + product + \" stock is low (\" + stockLevel + \" remaining)\");\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} system = new {name}();\n        Observer supplier1 = new Supplier(\"Electronics Warehouse\");\n        Observer supplier2 = new Supplier(\"Grocery Supplier\");\n        \n        system.addObserver(supplier1);\n        system.addObserver(supplier2);\n        \n        system.updateStock(\"Laptops\", 5);\n        system.updateStock(\"Apples\", 10);\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "AI-driven inventory tracking system alerts suppliers when stock is low",
        "edge_cases": [
            "Handling bulk orders depleting stock rapidly",
            "Avoiding duplicate order notifications"
        ],
        "dependencies": [
            "IoT shelf monitoring, AI demand forecasting"
        ],
        "performance_notes": "Optimized for automated restocking and demand prediction",
        "real_world_usage": "Used in retail stores like Walmart, Amazon Fresh",
        "testing_notes": "Simulate different product demand levels and supplier responses",
        "comments": "Can integrate with AI-driven predictive supply chain management",
        "source": "Inspired by AI-powered retail automation"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class Tutor implements Observer {{\n    private String name;\n    \n    public Tutor(String name) {{\n        this.name = name;\n    }}\n    \n    @Override\n    public void update(String student, String subject, String progress) {{\n        System.out.println(\"ALERT for \" + name + \": \" + student + \"'s progress in \" + subject + \" is \" + progress);\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} platform = new {name}();\n        Observer tutor1 = new Tutor(\"Mr. Smith\");\n        Observer tutor2 = new Tutor(\"Mrs. Johnson\");\n        \n        platform.addObserver(tutor1);\n        platform.addObserver(tutor2);\n        \n        platform.updateProgress(\"Alice\", \"Math\", \"Excellent\");\n        platform.updateProgress(\"Bob\", \"Science\", \"Needs Improvement\");\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "AI-driven e-learning platform tracks student progress and alerts tutors",
        "edge_cases": [
            "Handling multiple subjects per student",
            "Ensuring unbiased AI assessment of student performance"
        ],
        "dependencies": [
            "AI-based learning models, student performance tracking systems"
        ],
        "performance_notes": "Optimized for personalized learning recommendations",
        "real_world_usage": "Used in platforms like Coursera, Khan Academy, Udemy",
        "testing_notes": "Simulate different learning patterns and tutor interventions",
        "comments": "Can be extended with AI-based automated tutoring suggestions",
        "source": "Inspired by AI-powered online education platforms"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class Doctor implements Observer {{\n    private String name;\n    \n    public Doctor(String name) {{\n        this.name = name;\n    }}\n    \n    @Override\n    public void update(String patient, String vitalSign, double value) {{\n        System.out.println(\"ALERT for Dr. \" + name + \": \" + patient + \"'s \" + vitalSign + \" is critical at \" + value);\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} system = new {name}();\n        Observer doctor1 = new Doctor(\"Smith\");\n        Observer doctor2 = new Doctor(\"Jones\");\n        \n        system.addObserver(doctor1);\n        system.addObserver(doctor2);\n        \n        system.monitorVitals(\"John Doe\", \"Heart Rate\", 180);\n        system.monitorVitals(\"Jane Doe\", \"Blood Pressure\", 90);\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "AI-driven patient monitoring system alerts doctors about critical vitals",
        "edge_cases": [
            "Avoiding false alarms for minor fluctuations",
            "Ensuring real-time alerts for emergencies"
        ],
        "dependencies": [
            "IoT medical devices, AI-based anomaly detection"
        ],
        "performance_notes": "Optimized for real-time patient monitoring",
        "real_world_usage": "Used in ICU and emergency monitoring systems",
        "testing_notes": "Simulate various patient health conditions",
        "comments": "Can integrate with wearable health monitors for real-time tracking",
        "source": "Inspired by AI-driven remote patient monitoring solutions"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class MarketingAgency implements Observer {{\n    private String agencyName;\n    \n    public MarketingAgency(String agencyName) {{\n        this.agencyName = agencyName;\n    }}\n    \n    @Override\n    public void update(String topic, int mentions) {{\n        System.out.println(\"ALERT for \" + agencyName + \": Trending topic - \" + topic + \" with \" + mentions + \" mentions\");\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} analyzer = new {name}();\n        Observer agency1 = new MarketingAgency(\"Ad Agency A\");\n        Observer agency2 = new MarketingAgency(\"Brand Insights B\");\n        \n        analyzer.addObserver(agency1);\n        analyzer.addObserver(agency2);\n        \n        analyzer.detectTrend(\"New iPhone\", 10000);\n        analyzer.detectTrend(\"Tesla Stock Surge\", 8000);\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "AI-driven social media monitoring system alerts businesses about trending topics",
        "edge_cases": [
            "Filtering spam trends",
            "Detecting artificially boosted trends"
        ],
        "dependencies": [
            "AI-based sentiment analysis, social media APIs"
        ],
        "performance_notes": "Optimized for real-time social media data processing",
        "real_world_usage": "Used in marketing analytics and brand reputation management",
        "testing_notes": "Simulate different viral trends and influencer mentions",
        "comments": "Can integrate with AI-based content marketing strategies",
        "source": "Inspired by AI-driven social media analytics"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class SecurityAnalyst implements Observer {{\n    private String name;\n    \n    public SecurityAnalyst(String name) {{\n        this.name = name;\n    }}\n    \n    @Override\n    public void update(String threatType, String sourceIP) {{\n        System.out.println(\"ALERT for \" + name + \": \" + threatType + \" detected from IP \" + sourceIP);\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} system = new {name}();\n        Observer analyst1 = new SecurityAnalyst(\"CyberSec Team A\");\n        Observer analyst2 = new SecurityAnalyst(\"SOC Unit B\");\n        \n        system.addObserver(analyst1);\n        system.addObserver(analyst2);\n        \n        system.detectThreat(\"DDoS Attack\", \"192.168.1.10\");\n        system.detectThreat(\"SQL Injection\", \"203.45.67.89\");\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "AI-powered threat detection system alerts security analysts of cyber attacks",
        "edge_cases": [
            "Minimizing false positives",
            "Handling multiple concurrent attacks"
        ],
        "dependencies": [
            "AI-based anomaly detection models, SIEM tools"
        ],
        "performance_notes": "Optimized for real-time threat detection",
        "real_world_usage": "Used in cybersecurity defense systems like FireEye, CrowdStrike",
        "testing_notes": "Simulate different cyber attack scenarios",
        "comments": "Can integrate with AI-driven automated threat response systems",
        "source": "Inspired by AI-driven cybersecurity solutions"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class PowerStation implements Observer {{\n    private String name;\n    \n    public PowerStation(String name) {{\n        this.name = name;\n    }}\n    \n    @Override\n    public void update(String region, double demand) {{\n        System.out.println(\"Power Station \" + name + \" balancing load for \" + region + \" - Demand: \" + demand + \" MW\");\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} grid = new {name}();\n        Observer station1 = new PowerStation(\"Hydro Plant A\");\n        Observer station2 = new PowerStation(\"Solar Farm B\");\n        \n        grid.addObserver(station1);\n        grid.addObserver(station2);\n        \n        grid.reportDemand(\"New York\", 3000);\n        grid.reportDemand(\"Los Angeles\", 4500);\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "AI-driven smart grid dynamically adjusts energy distribution",
        "edge_cases": [
            "Handling sudden demand spikes",
            "Preventing blackouts during peak hours"
        ],
        "dependencies": [
            "IoT smart meters, AI demand forecasting"
        ],
        "performance_notes": "Optimized for real-time power redistribution",
        "real_world_usage": "Used in smart energy grids like Tesla Powerwall, GridOS",
        "testing_notes": "Simulate different energy demand fluctuations",
        "comments": "Can integrate with renewable energy sources like wind and solar",
        "source": "Inspired by AI-driven smart city infrastructure"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class SecurityTeam implements Observer {{\n    private String name;\n    \n    public SecurityTeam(String name) {{\n        this.name = name;\n    }}\n    \n    @Override\n    public void update(String transactionID, String alertType) {{\n        System.out.println(\"ALERT for \" + name + \": Suspicious transaction \" + transactionID + \" - \" + alertType);\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} system = new {name}();\n        Observer team1 = new SecurityTeam(\"Bank Security\");\n        Observer team2 = new SecurityTeam(\"E-Commerce Risk Department\");\n        \n        system.addObserver(team1);\n        system.addObserver(team2);\n        \n        system.detectFraud(\"TXN78901\", \"High-value transaction from unusual location\");\n        system.detectFraud(\"TXN45678\", \"Multiple failed login attempts\");\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "AI-powered fraud detection system monitors e-commerce transactions",
        "edge_cases": [
            "Minimizing false alarms",
            "Handling large-scale transaction data"
        ],
        "dependencies": [
            "AI-based fraud detection algorithms, payment processing APIs"
        ],
        "performance_notes": "Optimized for real-time fraud monitoring",
        "real_world_usage": "Used in payment gateways like PayPal, Stripe, Visa",
        "testing_notes": "Simulate different fraudulent activity scenarios",
        "comments": "Can integrate with AI-based fraud prevention models",
        "source": "Inspired by AI-driven financial security solutions"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class User implements Observer {{\n    private String name;\n    \n    public User(String name) {{\n        this.name = name;\n    }}\n    \n    @Override\n    public void update(String user, String recommendedSong) {{\n        if (this.name.equals(user)) {{\n            System.out.println(\"Hey \" + user + \"! We recommend you listen to: \" + recommendedSong);\n        }}\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} system = new {name}();\n        Observer user1 = new User(\"Alice\");\n        Observer user2 = new User(\"Bob\");\n        \n        system.addObserver(user1);\n        system.addObserver(user2);\n        \n        system.generateRecommendation(\"Alice\", \"Blinding Lights - The Weeknd\");\n        system.generateRecommendation(\"Bob\", \"Bohemian Rhapsody - Queen\");\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "AI-powered music recommendation system tracks user listening habits and suggests songs",
        "edge_cases": [
            "Ensuring diverse song recommendations",
            "Avoiding repetition of previous recommendations"
        ],
        "dependencies": [
            "Machine learning-based recommendation algorithms, music metadata APIs"
        ],
        "performance_notes": "Optimized for real-time user behavior tracking",
        "real_world_usage": "Used in streaming services like Spotify, Apple Music, YouTube Music",
        "testing_notes": "Simulate different user listening patterns",
        "comments": "Can integrate with AI-based sentiment analysis for mood-based recommendations",
        "source": "Inspired by AI-driven music streaming personalization"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class Resident implements Observer {{\n    private String name;\n    \n    public Resident(String name) {{\n        this.name = name;\n    }}\n    \n    @Override\n    public void update(String location, String weatherAlert) {{\n        System.out.println(\"ALERT for \" + name + \" in \" + location + \": \" + weatherAlert);\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} system = new {name}();\n        Observer resident1 = new Resident(\"John\");\n        Observer resident2 = new Resident(\"Emily\");\n        \n        system.addObserver(resident1);\n        system.addObserver(resident2);\n        \n        system.sendWeatherAlert(\"California\", \"Severe Storm Warning\");\n        system.sendWeatherAlert(\"Florida\", \"Hurricane Alert\");\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "AI-powered weather alert system notifies users about severe conditions",
        "edge_cases": [
            "Ensuring timely notifications",
            "Minimizing false alerts"
        ],
        "dependencies": [
            "AI-based weather prediction models, real-time meteorological data"
        ],
        "performance_notes": "Optimized for real-time climate monitoring",
        "real_world_usage": "Used in weather forecasting apps and emergency alert systems",
        "testing_notes": "Simulate different weather events and user notifications",
        "comments": "Can integrate with IoT-based environmental sensors",
        "source": "Inspired by AI-driven meteorological systems"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class Bidder implements Observer {{\n    private String name;\n    \n    public Bidder(String name) {{\n        this.name = name;\n    }}\n    \n    @Override\n    public void update(String item, double highestBid) {{\n        System.out.println(\"ALERT for \" + name + \": \" + item + \" now has a highest bid of $\" + highestBid);\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} system = new {name}();\n        Observer bidder1 = new Bidder(\"Alice\");\n        Observer bidder2 = new Bidder(\"Bob\");\n        \n        system.addObserver(bidder1);\n        system.addObserver(bidder2);\n        \n        system.newBid(\"Rare Painting\", 5000.0);\n        system.newBid(\"Vintage Car\", 12000.0);\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "AI-powered auction system updates bidders on real-time bids",
        "edge_cases": [
            "Handling last-minute bid updates",
            "Preventing fake bid placements"
        ],
        "dependencies": [
            "AI-based fraud detection in auction bidding"
        ],
        "performance_notes": "Optimized for high-frequency bid updates",
        "real_world_usage": "Used in online auction platforms like eBay, Christie's",
        "testing_notes": "Simulate multiple users bidding simultaneously",
        "comments": "Can integrate with AI-based price prediction for bidding strategies",
        "source": "Inspired by AI-driven online bidding systems"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class DroneOperator implements Observer {{\n    private String name;\n    \n    public DroneOperator(String name) {{\n        this.name = name;\n    }}\n    \n    @Override\n    public void update(String droneID, String status) {{\n        System.out.println(\"ALERT for \" + name + \": Drone \" + droneID + \" status: \" + status);\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} system = new {name}();\n        Observer operator1 = new DroneOperator(\"John\");\n        Observer operator2 = new DroneOperator(\"Emily\");\n        \n        system.addObserver(operator1);\n        system.addObserver(operator2);\n        \n        system.updateFlightStatus(\"DRN1001\", \"Battery Low - Returning to Base\");\n        system.updateFlightStatus(\"DRN2005\", \"Delivery Completed\");\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "AI-powered drone management system monitors and alerts operators about autonomous flight updates",
        "edge_cases": [
            "Handling signal loss or GPS malfunctions",
            "Coordinating multiple drones simultaneously"
        ],
        "dependencies": [
            "AI-based flight path optimization, real-time GPS tracking"
        ],
        "performance_notes": "Optimized for high-speed drone coordination",
        "real_world_usage": "Used in logistics, aerial surveillance, emergency response",
        "testing_notes": "Simulate different drone flight disruptions",
        "comments": "Can integrate with AI-driven collision avoidance systems",
        "source": "Inspired by AI-powered autonomous drone fleets"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class Bettor implements Observer {{\n    private String name;\n    \n    public Bettor(String name) {{\n        this.name = name;\n    }}\n    \n    @Override\n    public void update(String game, String event, double odds) {{\n        System.out.println(\"ALERT for \" + name + \": \" + game + \" - \" + event + \" (Odds: \" + odds + \")\");\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} platform = new {name}();\n        Observer bettor1 = new Bettor(\"Jake\");\n        Observer bettor2 = new Bettor(\"Lisa\");\n        \n        platform.addObserver(bettor1);\n        platform.addObserver(bettor2);\n        \n        platform.notifyEvent(\"NBA Finals\", \"Player Scored\", 1.8);\n        platform.notifyEvent(\"Champions League\", \"Goal Scored\", 2.5);\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "AI-driven betting platform updates users about live sports betting opportunities",
        "edge_cases": [
            "Ensuring real-time updates for high-speed games",
            "Avoiding duplicate notifications for similar events"
        ],
        "dependencies": [
            "Real-time sports analytics, AI-based odds calculation"
        ],
        "performance_notes": "Optimized for ultra-fast odds updates",
        "real_world_usage": "Used in betting platforms like DraftKings, FanDuel, Bet365",
        "testing_notes": "Simulate different game events and betting trends",
        "comments": "Can integrate with AI-based risk assessment models for betting strategies",
        "source": "Inspired by AI-driven sports betting analytics"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class Employee implements Observer {{\n    private String name;\n    \n    public Employee(String name) {{\n        this.name = name;\n    }}\n    \n    @Override\n    public void update(String room, String status) {{\n        System.out.println(\"ALERT for \" + name + \": Meeting room \" + room + \" is now \" + status);\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} system = new {name}();\n        Observer employee1 = new Employee(\"David\");\n        Observer employee2 = new Employee(\"Sophia\");\n        \n        system.addObserver(employee1);\n        system.addObserver(employee2);\n        \n        system.updateRoomStatus(\"Conference Room A\", \"Available\");\n        system.updateRoomStatus(\"Huddle Room 2\", \"Booked\");\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "AI-powered smart office system tracks and notifies employees about room bookings",
        "edge_cases": [
            "Handling last-minute booking cancellations",
            "Avoiding overbooking of rooms"
        ],
        "dependencies": [
            "IoT sensors for room occupancy, AI-based booking recommendations"
        ],
        "performance_notes": "Optimized for real-time meeting room scheduling",
        "real_world_usage": "Used in office management software like Google Workspace, Microsoft 365",
        "testing_notes": "Simulate different room booking scenarios",
        "comments": "Can integrate with AI-driven office automation for energy savings",
        "source": "Inspired by AI-powered smart workspace solutions"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class PoliceDepartment implements Observer {{\n    private String name;\n    \n    public PoliceDepartment(String name) {{\n        this.name = name;\n    }}\n    \n    @Override\n    public void update(String vehicle, String violation, String location) {{\n        System.out.println(\"ALERT for \" + name + \": Vehicle \" + vehicle + \" committed \" + violation + \" at \" + location);\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} system = new {name}();\n        Observer police1 = new PoliceDepartment(\"City PD\");\n        Observer police2 = new PoliceDepartment(\"Highway Patrol\");\n        \n        system.addObserver(police1);\n        system.addObserver(police2);\n        \n        system.reportViolation(\"ABC-123\", \"Speeding\", \"Main Street\");\n        system.reportViolation(\"XYZ-789\", \"Red Light Violation\", \"5th Avenue\");\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "AI-powered traffic monitoring system detects and reports traffic violations",
        "edge_cases": [
            "Avoiding false detections due to camera malfunctions",
            "Minimizing duplicate violation reports for frequent offenders"
        ],
        "dependencies": [
            "AI-based image recognition, IoT-based speed tracking"
        ],
        "performance_notes": "Optimized for real-time traffic monitoring",
        "real_world_usage": "Used in smart city infrastructure and automated traffic enforcement",
        "testing_notes": "Simulate different traffic violation scenarios",
        "comments": "Can integrate with AI-driven predictive analytics for traffic trends",
        "source": "Inspired by AI-powered traffic law enforcement systems"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class Subscriber implements Observer {{\n    private String name;\n    \n    public Subscriber(String name) {{\n        this.name = name;\n    }}\n    \n    @Override\n    public void update(String topic, String headline) {{\n        System.out.println(\"Hey \" + name + \"! New update on \" + topic + \": \" + headline);\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} system = new {name}();\n        Observer subscriber1 = new Subscriber(\"Alice\");\n        Observer subscriber2 = new Subscriber(\"Bob\");\n        \n        system.addObserver(subscriber1);\n        system.addObserver(subscriber2);\n        \n        system.publishNews(\"Technology\", \"New AI Model Released by OpenAI\");\n        system.publishNews(\"Finance\", \"Stock Market Hits Record High\");\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "AI-powered news aggregation system tracks trending stories and alerts users",
        "edge_cases": [
            "Filtering out fake news",
            "Minimizing duplicate notifications for frequent updates"
        ],
        "dependencies": [
            "AI-based sentiment analysis, news APIs"
        ],
        "performance_notes": "Optimized for personalized news recommendations",
        "real_world_usage": "Used in news platforms like Google News, Flipboard, Apple News",
        "testing_notes": "Simulate different user preferences and news updates",
        "comments": "Can integrate with AI-based topic modeling for personalized alerts",
        "source": "Inspired by AI-powered media intelligence"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class Farmer implements Observer {{\n    private String name;\n    \n    public Farmer(String name) {{\n        this.name = name;\n    }}\n    \n    @Override\n    public void update(String field, String issue) {{\n        System.out.println(\"ALERT for \" + name + \": Crop issue detected in \" + field + \" - \" + issue);\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} system = new {name}();\n        Observer farmer1 = new Farmer(\"John\");\n        Observer farmer2 = new Farmer(\"Emily\");\n        \n        system.addObserver(farmer1);\n        system.addObserver(farmer2);\n        \n        system.detectIssue(\"North Field\", \"Fungal Infection Detected\");\n        system.detectIssue(\"South Field\", \"Soil Nutrient Deficiency\");\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "AI-powered smart farming system tracks crop health and alerts farmers",
        "edge_cases": [
            "Handling false positive alerts",
            "Minimizing redundant notifications for ongoing issues"
        ],
        "dependencies": [
            "IoT-based soil sensors, AI-based crop disease detection"
        ],
        "performance_notes": "Optimized for real-time crop health analysis",
        "real_world_usage": "Used in precision agriculture and automated farming",
        "testing_notes": "Simulate different weather conditions and crop diseases",
        "comments": "Can integrate with AI-powered irrigation control",
        "source": "Inspired by AI-driven agricultural technology"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class MaintenanceEngineer implements Observer {{\n    private String name;\n    \n    public MaintenanceEngineer(String name) {{\n        this.name = name;\n    }}\n    \n    @Override\n    public void update(String machine, String warning) {{\n        System.out.println(\"ALERT for \" + name + \": Machine \" + machine + \" warning - \" + warning);\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} system = new {name}();\n        Observer engineer1 = new MaintenanceEngineer(\"Alex\");\n        Observer engineer2 = new MaintenanceEngineer(\"Lisa\");\n        \n        system.addObserver(engineer1);\n        system.addObserver(engineer2);\n        \n        system.detectWarning(\"CNC Machine\", \"Overheating detected\");\n        system.detectWarning(\"Conveyor Belt\", \"Vibration anomaly detected\");\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "AI-powered predictive maintenance system alerts engineers about potential machine failures",
        "edge_cases": [
            "Handling minor fluctuations vs. critical failures",
            "Reducing false alerts to avoid unnecessary downtime"
        ],
        "dependencies": [
            "IoT sensors for machine condition tracking, AI-based failure prediction"
        ],
        "performance_notes": "Optimized for real-time industrial monitoring",
        "real_world_usage": "Used in Industry 4.0 smart factories",
        "testing_notes": "Simulate different machine failure scenarios",
        "comments": "Can be integrated with AI-based preventive maintenance scheduling",
        "source": "Inspired by AI-driven predictive maintenance solutions"
    },
    {
        "type": "Observer",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "class Student implements Observer {{\n    private String name;\n    \n    public Student(String name) {{\n        this.name = name;\n    }}\n    \n    @Override\n    public void update(String student, String recommendedPlan) {{\n        if (this.name.equals(student)) {{\n            System.out.println(\"Hey \" + student + \"! Your new recommended study plan: \" + recommendedPlan);\n        }}\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} system = new {name}();\n        Observer student1 = new Student(\"Alice\");\n        Observer student2 = new Student(\"Bob\");\n        \n        system.addObserver(student1);\n        system.addObserver(student2);\n        \n        system.recommendStudyPlan(\"Alice\", \"Focus on Math and Physics\");\n        system.recommendStudyPlan(\"Bob\", \"Improve writing and reading comprehension\");\n    }}\n}}",
        "complexity": "Advanced",
        "language": "Java",
        "context": "AI-powered personalized learning platform recommends study plans based on student performance",
        "edge_cases": [
            "Handling inconsistent study patterns",
            "Avoiding redundant recommendations"
        ],
        "dependencies": [
            "AI-based learning models, student performance tracking systems"
        ],
        "performance_notes": "Optimized for personalized learning",
        "real_world_usage": "Used in platforms like Coursera, Udemy, Khan Academy",
        "testing_notes": "Simulate different student learning behaviors",
        "comments": "Can integrate with AI-powered tutoring assistance",
        "source": "Inspired by AI-driven personalized education solutions"
    }

]

# Generate examples for each pattern
examples = []
unique_checker = set()  # To track unique (input, output) pairs

for pattern in patterns:
    for i in range(10):  # Generate 5 variations per pattern
        class_name = f"{pattern['type']}Example{i + 1}"
        generated_input = pattern["input_template"].format(name=class_name)
        generated_output = pattern["output_template"].format(name=class_name)

        # Check if the (input, output) pair is unique
        if (generated_input, generated_output) not in unique_checker:
            examples.append({
                "type": pattern["type"],
                "input": generated_input,
                "output": generated_output,
                "complexity": pattern["complexity"],
                "language": pattern["language"],
                "context": pattern["context"],
                "edge_cases": pattern["edge_cases"],
                "dependencies": pattern["dependencies"],
                "performance_notes": pattern["performance_notes"],
                "real_world_usage": pattern["real_world_usage"],
                "testing_notes": pattern["testing_notes"],
                "comments": pattern["comments"],
                "source": pattern["source"]
            })
            unique_checker.add((generated_input, generated_output))  # Mark as generated

# Save the dataset to a JSON file
dataset_filename = "augmented_Observer_method_data.json"
with open(dataset_filename, "w") as json_file:
    json.dump(examples, json_file, indent=4)

print(f"Dataset generated and saved to {dataset_filename}.")
