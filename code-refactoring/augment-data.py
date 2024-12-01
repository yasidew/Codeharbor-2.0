import json

# Patterns for Singleton types with placeholders
patterns = [
    {
        "type": "EagerlyInitializedSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private {name}() {{}}\n\n    private static final {name} instance = new {name}();\n\n    public static {name} getInstance() {{\n        return instance;\n    }}\n\n}}"
    },
    {
        "type": "LazyInitializedSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n\n    private {name}() {{}}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n}}"
    },
    {
        "type": "ThreadSafeSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n\n    private {name}() {{}}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n}}"
    },
    {
        "type": "BillPughSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private {name}() {{}}\n\n    private static class SingletonHelper {{\n        private static final {name} INSTANCE = new {name}();\n    }}\n\n    public static {name} getInstance() {{\n        return SingletonHelper.INSTANCE;\n    }}\n\n}}"
    },
    {
        "type": "SerializableSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.io.Serializable;\n\npublic class {name} implements Serializable {{\n\n    private static final long serialVersionUID = 1L;\n    private static {name} instance;\n\n    private {name}() {{}}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    protected Object readResolve() {{\n        return getInstance();\n    }}\n\n}}"
    },
    {
        "type": "DoubleCheckedLockingSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static volatile {name} instance;\n\n    private {name}() {{}}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            synchronized ({name}.class) {{\n                if (instance == null) {{\n                    instance = new {name}();\n                }}\n            }}\n        }}\n        return instance;\n    }}\n\n}}"
    },
    {
        "type": "StaticBlockSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n\n    static {{\n        try {{\n            instance = new {name}();\n        }} catch (Exception e) {{\n            e.printStackTrace();\n        }}\n    }}\n\n    private {name}() {{}}\n\n    public static {name} getInstance() {{\n        return instance;\n    }}\n\n}}"
    },
    {
        "type": "ObserverPatternSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.ArrayList;\nimport java.util.List;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private List<Observer> observers;\n\n    private {name}() {{\n        observers = new ArrayList<>();\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void addObserver(Observer observer) {{\n        observers.add(observer);\n    }}\n\n    public void notifyObservers() {{\n        for (Observer observer : observers) {{\n            observer.update();\n        }}\n    }}\n\n}}\n\ninterface Observer {{\n    void update();\n}}"
    },
    {
        "type": "ThreadLocalSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static final ThreadLocal<{name}> threadLocalInstance = ThreadLocal.withInitial({name}::new);\n\n    private {name}() {{}}\n\n    public static {name} getInstance() {{\n        return threadLocalInstance.get();\n    }}\n\n}}"
    },
{
        "type": "EagerlyInitializedSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private {name}() {{}}\n\n    private static final {name} instance = new {name}();\n\n    public static {name} getInstance() {{\n        return instance;\n    }}\n\n}}"
    },
    {
        "type": "LazyInitializedSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n\n    private {name}() {{}}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n}}"
    },
    {
        "type": "ThreadSafeSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n\n    private {name}() {{}}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n}}"
    },
    {
        "type": "BillPughSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private {name}() {{}}\n\n    private static class SingletonHelper {{\n        private static final {name} INSTANCE = new {name}();\n    }}\n\n    public static {name} getInstance() {{\n        return SingletonHelper.INSTANCE;\n    }}\n\n}}"
    },
    {
        "type": "EnumSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public enum {name} {{\n    INSTANCE;\n\n    public void someMethod() {{\n        // Your method implementation\n    }}\n}}"
    },
    {
        "type": "EagerlyInitializedStaticBlockSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n\n    private {name}() {{}}\n\n    static {{\n        try {{\n            instance = new {name}();\n        }} catch (Exception ex) {{\n            ex.printStackTrace();\n        }}\n    }}\n\n    public static {name} getInstance() {{\n        return instance;\n    }}\n\n}}"
    },
    {
        "type": "EnumSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public enum {name} {{\n\n    INSTANCE;\n\n    //other\n}}"
    },
    {
        "type": "LazilyInitializedDoubleCheckedLockingSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static volatile {name} instance;\n\n    private {name}() {{}}\n\n    public static {name} getInstance() {{\n        if(instance == null) {{\n            synchronized ({name}.class) {{\n                if(instance == null) {{\n                    instance = new {name}();\n                }}\n            }}\n        }}\n        return instance;\n    }}\n\n}}"
    },
    {
        "type": "LazilyInitializedInnerClassSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private {name}() {{}}\n\n    private static class InnerSingletonInitializer {{\n        private static final {name} INSTANCE = new {name}();\n    }}\n\n    public static {name} getInstance() {{\n        return InnerSingletonInitializer.INSTANCE;\n    }}\n\n}}"
    },
    {
        "type": "LazilyInitializedSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n\n    private {name}() {{}}\n\n    public static synchronized {name} getInstance() {{\n        if(instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n}}"
    },
    {
        "type": "ProtectionAgainstReflectionSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private {name}() {{\n        if(instance != null) {{\n            throw new IllegalStateException(\"Singleton already initialized\");\n        }}\n    }}\n\n    private static final {name} instance = new {name}();\n\n    public static {name} getInstance() {{\n        return instance;\n    }}\n\n}}"
    },
    {
        "type": "SerializableSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} implements Serializable {{\n\n    private static final long serialVersionUID = -6265755052204900542L;\n\n    private static {name} instance;\n\n    private {name}() {{}}\n\n    public static synchronized {name} getInstance() {{\n        if(instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n}}"
    },
    {
        "type": "SerializableWithReadResolveSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} implements Serializable {{\n\n    private static final long serialVersionUID = 1911904003687931976L;\n\n    private static {name} instance;\n\n    private {name}() {{}}\n\n    protected Object readResolve() {{\n        return instance;\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if(instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n}}"
    },
    {
        "type": "LazilyInitializedSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n    private static {name} instance;\n\n    private String name;\n    private int numberOfGalaxies;\n\n    private {name}(String name, int numberOfGalaxies) {{\n        this.name = name;\n        this.numberOfGalaxies = numberOfGalaxies;\n    }}\n\n    public static {name} getInstance(String name, int numberOfGalaxies) {{\n        if (instance == null) {{\n            instance = new {name}(name, numberOfGalaxies);\n        }}\n        return instance;\n    }}\n\n    public String getName() {{\n        return this.name;\n    }}\n\n    public int getNumberOfGalaxies() {{\n        return this.numberOfGalaxies;\n    }}\n\n    public void setName(String aNewName) {{\n        this.name = aNewName;\n    }}\n}}\n\npublic class BigBang {{\n    public BigBang() {{}}\n\n    public {name} make{name}(String name, int numberOfGalaxies) {{\n        return {name}.getInstance(name, numberOfGalaxies);\n    }}\n}}"
    },
    {
        "type": "LazilyInitializedSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n    private static {name} _captain;\n\n    private {name}() {{}}\n\n    public static {name} getCaptain() {{\n        if (_captain == null) {{\n            _captain = new {name}();\n            System.out.println(\"Izabran novi kapiten\");\n        }} else {{\n            System.out.println(\"Vec postoji kapiten\");\n        }}\n        return _captain;\n    }}\n\n    public static void main(String[] args) {{\n        {name} c1 = {name}.getCaptain();\n        {name} c2 = {name}.getCaptain();\n\n        if (c1 == c2) {{\n            System.out.println(\"c1 i c2 su iste instance\");\n        }}\n    }}\n}}"
    },
    {
        "type": "DoubleCheckedLockingSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n  private static volatile {name} publicPrinter;\n  private static volatile Connection connection;\n\n  private {name}() {{\n    if (publicPrinter != null) {{\n      throw new RuntimeException(\"please use publicPrinter\");\n    }}\n  }}\n\n  public static {name} getPublicPrinter() {{\n    if (publicPrinter == null) {{\n      synchronized ({name}.class) {{\n        if (publicPrinter == null) {{\n          publicPrinter = new {name}();\n        }}\n      }}\n    }}\n    return publicPrinter;\n  }}\n\n  public Connection getConnection() {{\n    if (connection == null) {{\n      synchronized ({name}.class) {{\n        if (connection == null) {{\n          String url = \"jdbc:derby:memory:sample;create=true\";\n          try {{\n            connection = DriverManager.getConnection(url);\n          }} catch (SQLException e) {{\n            e.printStackTrace();\n          }}\n        }}\n      }}\n    }}\n    return connection;\n  }}\n}}"
    },
    {
        "type": "LazilyInitializedSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    public String str;\n\n    private {name}() {{\n        str = \"This is a string of singleton\";\n    }}\n\n    private static {name} singleInstance = null;\n\n    public static {name} getInstance() {{\n        if (singleInstance == null) {{\n            singleInstance = new {name}();\n        }}\n        return singleInstance;\n    }}\n\n    public static void main(String[] args) {{\n        {name} s = {name}.getInstance();\n\n        {name} t = {name}.getInstance();\n\n        {name} u = {name}.getInstance();\n\n        t.str = t.str.toUpperCase();\n\n        System.out.println(\"For Instance of 's': \" + s.str);\n        System.out.println(\"For Instance of 't': \" + t.str);\n        System.out.println(\"For Instance of 'u': \" + u.str);\n    }}\n}}"
    },
    {
        "type": "DoubleCheckedLockingSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n    private static volatile {name} instance = null;\n\n    private {name}() {{\n        // private constructor\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            synchronized ({name}.class) {{\n                if (instance == null) {{\n                    instance = new {name}();\n                }}\n            }}\n        }}\n        return instance;\n    }}\n}}"
    },
    {
        "type": "EagerSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n    private static final {name} instance = new {name}();\n\n    private {name}() {{\n        // private constructor\n    }}\n\n    public static {name} getInstance() {{\n        return instance;\n    }}\n}}"
    },
    {
        "type": "EnumSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public enum {name} {{\n\n    INSTANCE(\"property want to initialize at once\");\n\n    private String info;\n\n    private {name}(String info) {{\n        this.info = info;\n    }}\n\n    public {name} getInstance() {{\n        return INSTANCE;\n    }}\n}}"
    },
    {
        "type": "LazySingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n    private static {name} instance;\n\n    private {name}() {{\n        // private constructor\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n}}"
    },
    {
        "type": "BillPughSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private {name}() {{\n    }}\n\n    private static class {name}Helper {{\n        private static final {name} INSTANCE = new {name}();\n    }}\n\n    public static {name} getInstance() {{\n        return {name}Helper.INSTANCE;\n    }}\n}}"
    },
    {
        "type": "StaticBlockSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n    public static {name} instance;\n\n    private {name}() {{\n        // private constructor\n    }}\n\n    static {{\n        instance = new {name}();\n    }}\n}}"
    },
    {
        "type": "ThreadSafeDoubleCheckedLockingSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n    private static volatile {name} instance = null;\n\n    private {name}() {{}}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            synchronized ({name}.class) {{\n                if (instance == null) {{\n                    instance = new {name}();\n                }}\n            }}\n        }}\n        return instance;\n    }}\n}}"
    },
    {
        "type": "OptimizedLazySingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static final Object lock = new Object();\n    private static volatile {name} instance;\n\n    public static {name} getInstance() {{\n        {name} r = instance;\n        if (r == null) {{\n            synchronized (lock) {{\n                r = instance;\n                if (r == null) {{\n                    r = new {name}();\n                    instance = r;\n                }}\n            }}\n        }}\n        return r;\n    }}\n}}"
    },
    {
        "type": "LazySingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n    \n    private static {name} instanciaUnica;\n\n    private {name}() {{\n        // private constructor\n    }}\n\n    public static {name} getInstancia() {{\n        if (instanciaUnica == null) {{\n            instanciaUnica = new {name}();\n        }}\n        return instanciaUnica;\n    }}\n\n    public void mostrarMensagem() {{\n        System.out.println(\"Exemplo de {name}!\");\n    }}\n}}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        {name} instancia = {name}.getInstancia();\n        instancia.mostrarMensagem();\n    }}\n}}"
    },
    {
        "type": "ThreadSafeSingletonWithInitializationOnDemand",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private {name}() {{}}\n\n    private static class SingletonHelper {{\n        private static final {name} INSTANCE = new {name}();\n    }}\n\n    public static {name} getInstance() {{\n        return SingletonHelper.INSTANCE;\n    }}\n\n}}"
    },
    {
        "type": "SynchronizedSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n\n    private {name}() {{}}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n}}"
    },
    {
        "type": "LazySingletonWithCounter",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n    private static int counter;\n\n    private {name}() {{\n        counter++;\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public int getInstanceCount() {{\n        return counter;\n    }}\n\n}}"
    },
    {
        "type": "EagerSingletonWithLogging",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static final {name} instance = new {name}();\n\n    private {name}() {{\n        System.out.println(\"Singleton instance created\");\n    }}\n\n    public static {name} getInstance() {{\n        return instance;\n    }}\n\n}}"
    },
    {
        "type": "EnumSingletonWithData",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public enum {name} {{\n\n    INSTANCE(\"default\");\n\n    private String data;\n\n    private {name}(String data) {{\n        this.data = data;\n    }}\n\n    public String getData() {{\n        return data;\n    }}\n\n    public void setData(String data) {{\n        this.data = data;\n    }}\n\n}}"
    },
    {
        "type": "StaticBlockSingletonWithErrorHandling",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n\n    private {name}() {{}}\n\n    static {{\n        try {{\n            instance = new {name}();\n        }} catch (Exception e) {{\n            throw new RuntimeException(\"Error creating singleton instance\", e);\n        }}\n    }}\n\n    public static {name} getInstance() {{\n        return instance;\n    }}\n\n}}"
    },
    {
        "type": "LazySingletonWithReadResolve",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} implements Serializable {{\n\n    private static final long serialVersionUID = 1L;\n    private static {name} instance;\n\n    private {name}() {{}}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    protected Object readResolve() {{\n        return getInstance();\n    }}\n\n}}"
    },
    {
        "type": "SingletonWithEnumAndMethod",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public enum {name} {{\n\n    INSTANCE;\n\n    public void performAction() {{\n        System.out.println(\"Action performed by {name}\");\n    }}\n\n}}"
    },
    {
        "type": "ThreadLocalSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static final ThreadLocal<{name}> instance = ThreadLocal.withInitial({name}::new);\n\n    private {name}() {{}}\n\n    public static {name} getInstance() {{\n        return instance.get();\n    }}\n\n}}"
    },
    {
        "type": "ParameterizedLazySingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n    private String param;\n\n    private {name}(String param) {{\n        this.param = param;\n    }}\n\n    public static synchronized {name} getInstance(String param) {{\n        if (instance == null) {{\n            instance = new {name}(param);\n        }}\n        return instance;\n    }}\n\n    public String getParam() {{\n        return param;\n    }}\n\n}}"
    },
    {
        "type": "LazySingletonWithInitializationCount",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n    private static int initializationCount = 0;\n\n    private {name}() {{\n        initializationCount++;\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public static int getInitializationCount() {{\n        return initializationCount;\n    }}\n\n}}"
    },
    {
        "type": "MultithreadedLazySingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n\n    private {name}() {{}}\n\n    public static {name} getInstance() {{\n        synchronized ({name}.class) {{\n            if (instance == null) {{\n                instance = new {name}();\n            }}\n        }}\n        return instance;\n    }}\n\n}}"
    },
    {
        "type": "LazySingletonWithOptionalInstance",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.Optional;\n\npublic class {name} {{\n\n    private static {name} instance;\n\n    private {name}() {{}}\n\n    public static Optional<{name}> getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return Optional.of(instance);\n    }}\n\n}}"
    },
    {
        "type": "LazySingletonWithDynamicConfig",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n    private String config;\n\n    private {name}(String config) {{\n        this.config = config;\n    }}\n\n    public static {name} getInstance(String config) {{\n        if (instance == null) {{\n            instance = new {name}(config);\n        }}\n        return instance;\n    }}\n\n    public String getConfig() {{\n        return config;\n    }}\n\n}}"
    },
    {
        "type": "EagerSingletonWithStartupLogging",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static final {name} instance = new {name}();\n\n    private {name}() {{\n        System.out.println(\"Singleton initialized during class loading\");\n    }}\n\n    public static {name} getInstance() {{\n        return instance;\n    }}\n\n}}"
    },
    {
        "type": "DoubleCheckedLockingWithLazyInitialization",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static volatile {name} instance;\n\n    private {name}() {{}}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            synchronized ({name}.class) {{\n                if (instance == null) {{\n                    instance = new {name}();\n                }}\n            }}\n        }}\n        return instance;\n    }}\n\n}}"
    },
    {
        "type": "SingletonWithWeakReference",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.lang.ref.WeakReference;\n\npublic class {name} {{\n\n    private static WeakReference<{name}> instance;\n\n    private {name}() {{}}\n\n    public static {name} getInstance() {{\n        if (instance == null || instance.get() == null) {{\n            instance = new WeakReference<>(new {name}());\n        }}\n        return instance.get();\n    }}\n\n}}"
    },
    {
        "type": "StaticBlockWithInstanceValidation",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n\n    private {name}() {{}}\n\n    static {{\n        try {{\n            instance = new {name}();\n        }} catch (Exception e) {{\n            System.err.println(\"Error during instance initialization: \" + e.getMessage());\n        }}\n    }}\n\n    public static {name} getInstance() {{\n        return instance;\n    }}\n\n}}"
    },
    {
        "type": "LazySingletonWithPropertiesFile",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.Properties;\nimport java.io.FileInputStream;\nimport java.io.IOException;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private Properties properties;\n\n    private {name}() {{\n        properties = new Properties();\n        try {{\n            properties.load(new FileInputStream(\"config.properties\"));\n        }} catch (IOException e) {{\n            e.printStackTrace();\n        }}\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public Properties getProperties() {{\n        return properties;\n    }}\n\n}}"
    },
    {
        "type": "SingletonWithCache",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.HashMap;\nimport java.util.Map;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private Map<String, String> cache;\n\n    private {name}() {{\n        cache = new HashMap<>();\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void putInCache(String key, String value) {{\n        cache.put(key, value);\n    }}\n\n    public String getFromCache(String key) {{\n        return cache.get(key);\n    }}\n\n}}"
    },
    {
        "type": "SingletonWithImmutableData",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n    private final String data;\n\n    private {name}(String data) {{\n        this.data = data;\n    }}\n\n    public static {name} getInstance(String data) {{\n        if (instance == null) {{\n            instance = new {name}(data);\n        }}\n        return instance;\n    }}\n\n    public String getData() {{\n        return data;\n    }}\n\n}}"
    },
    {
        "type": "SingletonWithBackgroundTask",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.concurrent.ExecutorService;\nimport java.util.concurrent.Executors;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private ExecutorService executorService;\n\n    private {name}() {{\n        executorService = Executors.newSingleThreadExecutor();\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void runTask(Runnable task) {{\n        executorService.submit(task);\n    }}\n\n}}"
    },
    {
        "type": "ClusteredSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.concurrent.ConcurrentHashMap;\nimport java.util.concurrent.ConcurrentMap;\n\npublic class {name} {{\n\n    private static final ConcurrentMap<String, {name}> instances = new ConcurrentHashMap<>();\n\n    private {name}() {{}}\n\n    public static {name} getInstance(String clusterKey) {{\n        return instances.computeIfAbsent(clusterKey, key -> new {name}());\n    }}\n\n}}"
    },
    {
        "type": "SingletonWithStateTracking",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n    private int state;\n\n    private {name}() {{\n        state = 0;\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void incrementState() {{\n        state++;\n    }}\n\n    public int getState() {{\n        return state;\n    }}\n\n}}"
    },
    {
        "type": "EnvironmentBasedSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n    private String environment;\n\n    private {name}(String environment) {{\n        this.environment = environment;\n    }}\n\n    public static {name} getInstance(String environment) {{\n        if (instance == null) {{\n            instance = new {name}(environment);\n        }}\n        return instance;\n    }}\n\n    public String getEnvironment() {{\n        return environment;\n    }}\n\n}}"
    },
    {
        "type": "SingletonWithThreadPool",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.concurrent.ExecutorService;\nimport java.util.concurrent.Executors;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private ExecutorService threadPool;\n\n    private {name}() {{\n        threadPool = Executors.newFixedThreadPool(5);\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public ExecutorService getThreadPool() {{\n        return threadPool;\n    }}\n\n}}"
    },
    {
        "type": "SingletonWithMultitonSupport",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.HashMap;\nimport java.util.Map;\n\npublic class {name} {{\n\n    private static Map<String, {name}> instances = new HashMap<>();\n\n    private {name}() {{}}\n\n    public static {name} getInstance(String key) {{\n        return instances.computeIfAbsent(key, k -> new {name}());\n    }}\n\n}}"
    },
    {
        "type": "SingletonWithObserverPattern",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.ArrayList;\nimport java.util.List;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private List<Observer> observers;\n\n    private {name}() {{\n        observers = new ArrayList<>();\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void addObserver(Observer observer) {{\n        observers.add(observer);\n    }}\n\n    public void notifyObservers() {{\n        for (Observer observer : observers) {{\n            observer.update();\n        }}\n    }}\n\n}}\n\ninterface Observer {{\n    void update();\n}}"
    },
    {
        "type": "SingletonWithLifecycleCallbacks",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n\n    private {name}() {{}}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n            onCreate();\n        }}\n        return instance;\n    }}\n\n    public static void onCreate() {{\n        System.out.println(\"Singleton instance created\");\n    }}\n\n    public static void onDestroy() {{\n        System.out.println(\"Singleton instance destroyed\");\n    }}\n\n}}"
    },
    {
        "type": "DatabaseConnectionSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.sql.Connection;\nimport java.sql.DriverManager;\nimport java.sql.SQLException;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private Connection connection;\n\n    private {name}() {{\n        try {{\n            connection = DriverManager.getConnection(\"jdbc:mysql://localhost:3306/db\", \"user\", \"password\");\n        }} catch (SQLException e) {{\n            e.printStackTrace();\n        }}\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public Connection getConnection() {{\n        return connection;\n    }}\n\n}}"
    },
    {
        "type": "LoggingSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.io.FileWriter;\nimport java.io.IOException;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private FileWriter fileWriter;\n\n    private {name}() {{\n        try {{\n            fileWriter = new FileWriter(\"app.log\", true);\n        }} catch (IOException e) {{\n            e.printStackTrace();\n        }}\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void log(String message) {{\n        try {{\n            fileWriter.write(message + \"\\n\");\n            fileWriter.flush();\n        }} catch (IOException e) {{\n            e.printStackTrace();\n        }}\n    }}\n\n}}"
    },
    {
        "type": "ConfigurationSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.Properties;\nimport java.io.FileInputStream;\nimport java.io.IOException;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private Properties properties;\n\n    private {name}() {{\n        properties = new Properties();\n        try {{\n            properties.load(new FileInputStream(\"config.properties\"));\n        }} catch (IOException e) {{\n            e.printStackTrace();\n        }}\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public String getConfig(String key) {{\n        return properties.getProperty(key);\n    }}\n\n}}"
    },
    {
        "type": "ThreadLocalSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static final ThreadLocal<{name}> threadLocalInstance = ThreadLocal.withInitial({name}::new);\n\n    private {name}() {{}}\n\n    public static {name} getInstance() {{\n        return threadLocalInstance.get();\n    }}\n\n}}"
    },
    {
        "type": "MultitonSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.HashMap;\nimport java.util.Map;\n\npublic class {name} {{\n\n    private static final Map<String, {name}> instances = new HashMap<>();\n\n    private {name}() {{}}\n\n    public static {name} getInstance(String key) {{\n        return instances.computeIfAbsent(key, k -> new {name}());\n    }}\n\n}}"
    },
    {
        "type": "SingletonWithSerialization",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.io.Serializable;\n\npublic class {name} implements Serializable {{\n\n    private static final long serialVersionUID = 1L;\n    private static {name} instance;\n\n    private {name}() {{}}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    protected Object readResolve() {{\n        return instance;\n    }}\n\n}}"
    },
    {
        "type": "LazyLoadingSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private {name}() {{}}\n\n    private static class Holder {{\n        private static final {name} INSTANCE = new {name}();\n    }}\n\n    public static {name} getInstance() {{\n        return Holder.INSTANCE;\n    }}\n\n}}"
    },
    {
        "type": "SingletonWithObserverPattern",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.ArrayList;\nimport java.util.List;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private List<Observer> observers;\n\n    private {name}() {{\n        observers = new ArrayList<>();\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void addObserver(Observer observer) {{\n        observers.add(observer);\n    }}\n\n    public void notifyObservers(String message) {{\n        for (Observer observer : observers) {{\n            observer.update(message);\n        }}\n    }}\n\n}}\n\ninterface Observer {{\n    void update(String message);\n}}"
    },
    {
        "type": "SingletonWithStateTracking",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n    private int state;\n\n    private {name}() {{\n        state = 0;\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void incrementState() {{\n        state++;\n    }}\n\n    public int getState() {{\n        return state;\n    }}\n\n}}"
    },
    {
        "type": "SingletonWithDynamicConfiguration",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n    private String config;\n\n    private {name}(String config) {{\n        this.config = config;\n    }}\n\n    public static {name} getInstance(String config) {{\n        if (instance == null) {{\n            instance = new {name}(config);\n        }}\n        return instance;\n    }}\n\n    public String getConfig() {{\n        return config;\n    }}\n\n}}"
    },
    {
        "type": "EventDrivenSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.ArrayList;\nimport java.util.List;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private List<String> eventQueue;\n\n    private {name}() {{\n        eventQueue = new ArrayList<>();\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void handleEvent(String event) {{\n        eventQueue.add(event);\n        System.out.println(\"Event handled: \" + event);\n    }}\n\n    public List<String> getEventQueue() {{\n        return eventQueue;\n    }}\n\n}}"
    },
    {
        "type": "SessionManagerSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.HashMap;\nimport java.util.Map;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private Map<String, String> activeSessions;\n\n    private {name}() {{\n        activeSessions = new HashMap<>();\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void createSession(String user) {{\n        activeSessions.put(user, \"SessionActive\");\n        System.out.println(\"Session created for: \" + user);\n    }}\n\n    public boolean isSessionActive(String user) {{\n        return activeSessions.containsKey(user);\n    }}\n\n}}"
    },
    {
        "type": "ConfigurationManagerSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.Properties;\nimport java.io.FileInputStream;\nimport java.io.IOException;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private Properties properties;\n\n    private {name}() {{\n        properties = new Properties();\n        try {{\n            properties.load(new FileInputStream(\"app_config.properties\"));\n        }} catch (IOException e) {{\n            e.printStackTrace();\n        }}\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public String getConfigValue(String key) {{\n        return properties.getProperty(key);\n    }}\n\n}}"
    },
    {
        "type": "ResourcePoolSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.Queue;\nimport java.util.LinkedList;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private Queue<String> availableResources;\n\n    private {name}() {{\n        availableResources = new LinkedList<>();\n        availableResources.add(\"Resource1\");\n        availableResources.add(\"Resource2\");\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public String allocateResource() {{\n        return availableResources.poll();\n    }}\n\n    public void releaseResource(String resource) {{\n        availableResources.add(resource);\n    }}\n\n}}"
    },
    {
        "type": "SingletonWithMetricsCollection",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.ArrayList;\nimport java.util.List;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private List<String> collectedMetrics;\n\n    private {name}() {{\n        collectedMetrics = new ArrayList<>();\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void collectMetric(String metric) {{\n        collectedMetrics.add(metric);\n    }}\n\n    public List<String> getMetrics() {{\n        return collectedMetrics;\n    }}\n\n}}"
    },
    {
        "type": "LazyLoadedSingletonWithLogging",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private {name}() {{\n        System.out.println(\"Singleton instance created\");\n    }}\n\n    private static class SingletonHolder {{\n        private static final {name} INSTANCE = new {name}();\n    }}\n\n    public static {name} getInstance() {{\n        return SingletonHolder.INSTANCE;\n    }}\n\n}}"
    },
    {
        "type": "PriorityBasedSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.PriorityQueue;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private PriorityQueue<Integer> priorityQueue;\n\n    private {name}() {{\n        priorityQueue = new PriorityQueue<>();\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void addPriority(int priority) {{\n        priorityQueue.add(priority);\n    }}\n\n    public int getHighestPriority() {{\n        return priorityQueue.poll();\n    }}\n\n}}"
    },
    {
        "type": "LazySingletonWithOnDemandInitialization",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n\n    private {name}() {{}}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public String fetchData() {{\n        return \"Fetched on-demand data\";\n    }}\n\n}}"
    },
    {
        "type": "SingletonWithLifecycleHooks",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n\n    private {name}() {{\n        onCreate();\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    private void onCreate() {{\n        System.out.println(\"{name} created\");\n    }}\n\n    public void onDestroy() {{\n        System.out.println(\"{name} destroyed\");\n        instance = null;\n    }}\n\n}}"
    },
    {
        "type": "LoggingManagerSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.io.FileWriter;\nimport java.io.IOException;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private FileWriter fileWriter;\n\n    private {name}() {{\n        try {{\n            fileWriter = new FileWriter(\"logfile.log\", true);\n        }} catch (IOException e) {{\n            e.printStackTrace();\n        }}\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void logMessage(String message) {{\n        try {{\n            fileWriter.write(message + \"\\n\");\n            fileWriter.flush();\n        }} catch (IOException e) {{\n            e.printStackTrace();\n        }}\n    }}\n\n}}"
    },
    {
        "type": "ResourceTrackerSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.HashSet;\nimport java.util.Set;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private Set<String> trackedResources;\n\n    private {name}() {{\n        trackedResources = new HashSet<>();\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void track(String resource) {{\n        trackedResources.add(resource);\n        System.out.println(\"Tracking: \" + resource);\n    }}\n\n    public boolean isTracked(String resource) {{\n        return trackedResources.contains(resource);\n    }}\n\n}}"
    },
    {
        "type": "TaskQueueSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.LinkedList;\nimport java.util.Queue;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private Queue<String> tasks;\n\n    private {name}() {{\n        tasks = new LinkedList<>();\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void addTask(String task) {{\n        tasks.add(task);\n    }}\n\n    public String getNextTask() {{\n        return tasks.poll();\n    }}\n\n}}"
    },
    {
        "type": "ThreadLocalLoggerSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static final ThreadLocal<{name}> threadLocalInstance = ThreadLocal.withInitial({name}::new);\n\n    private {name}() {{}}\n\n    public static {name} getInstance() {{\n        return threadLocalInstance.get();\n    }}\n\n    public void log(String message) {{\n        System.out.println(Thread.currentThread().getName() + \": \" + message);\n    }}\n\n}}"
    },
    {
        "type": "ConfigurationCacheSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.HashMap;\nimport java.util.Map;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private Map<String, String> cache;\n\n    private {name}() {{\n        cache = new HashMap<>();\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void put(String key, String value) {{\n        cache.put(key, value);\n    }}\n\n    public String get(String key) {{\n        return cache.get(key);\n    }}\n\n}}"
    },
    {
        "type": "EventDispatcherSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.ArrayList;\nimport java.util.List;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private List<String> events;\n\n    private {name}() {{\n        events = new ArrayList<>();\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void dispatch(String event) {{\n        events.add(event);\n        System.out.println(\"Event dispatched: \" + event);\n    }}\n\n    public List<String> getDispatchedEvents() {{\n        return events;\n    }}\n\n}}"
    },
    {
        "type": "UserManagerSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.HashMap;\nimport java.util.Map;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private Map<String, String> users;\n\n    private {name}() {{\n        users = new HashMap<>();\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void addUser(String id, String name) {{\n        users.put(id, name);\n    }}\n\n    public String getUser(String id) {{\n        return users.get(id);\n    }}\n\n}}"
    },
    {
        "type": "StatefulLoggerSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n    private int logCount;\n\n    private {name}() {{\n        logCount = 0;\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void log(String message) {{\n        logCount++;\n        System.out.println(logCount + \": \" + message);\n    }}\n\n    public int getLogCount() {{\n        return logCount;\n    }}\n\n}}"
    },
    {
        "type": "NotificationManagerSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.ArrayList;\nimport java.util.List;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private List<String> notifications;\n\n    private {name}() {{\n        notifications = new ArrayList<>();\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void notify(String message) {{\n        notifications.add(message);\n        System.out.println(\"Notification: \" + message);\n    }}\n\n    public List<String> getNotifications() {{\n        return notifications;\n    }}\n\n}}"
    },
    {
        "type": "LazySingletonWithRetry",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n    private int retryCount;\n\n    private {name}() {{\n        retryCount = 0;\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void retry(String task) {{\n        retryCount++;\n        System.out.println(\"Retrying task: \" + task + \" - Retry Count: \" + retryCount);\n    }}\n\n}}"
    },
    {
        "type": "DatabaseConnectionPoolSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.Queue;\nimport java.util.LinkedList;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private Queue<String> connectionPool;\n\n    private {name}() {{\n        connectionPool = new LinkedList<>();\n        connectionPool.add(\"Connection1\");\n        connectionPool.add(\"Connection2\");\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public String getConnection() {{\n        return connectionPool.poll();\n    }}\n\n    public void releaseConnection(String connection) {{\n        connectionPool.add(connection);\n    }}\n\n}}"
    },
    {
        "type": "CacheManagerSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.HashMap;\nimport java.util.Map;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private Map<String, String> cache;\n\n    private {name}() {{\n        cache = new HashMap<>();\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void put(String key, String value) {{\n        cache.put(key, value);\n    }}\n\n    public String get(String key) {{\n        return cache.get(key);\n    }}\n\n}}"
    },
    {
        "type": "MultithreadedTaskManagerSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.Queue;\nimport java.util.concurrent.ConcurrentLinkedQueue;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private Queue<String> taskQueue;\n\n    private {name}() {{\n        taskQueue = new ConcurrentLinkedQueue<>();\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void addTask(String task) {{\n        taskQueue.add(task);\n    }}\n\n    public String getTask() {{\n        return taskQueue.poll();\n    }}\n\n}}"
    },
    {
        "type": "ThreadSafeConfigManagerSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.concurrent.ConcurrentHashMap;\nimport java.util.Map;\n\npublic class {name} {{\n\n    private static volatile {name} instance;\n    private Map<String, String> config;\n\n    private {name}() {{\n        config = new ConcurrentHashMap<>();\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            synchronized ({name}.class) {{\n                if (instance == null) {{\n                    instance = new {name}();\n                }}\n            }}\n        }}\n        return instance;\n    }}\n\n    public void setConfig(String key, String value) {{\n        config.put(key, value);\n    }}\n\n    public String getConfig(String key) {{\n        return config.get(key);\n    }}\n\n}}"
    },
    {
        "type": "TokenBucketRateLimiterSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.concurrent.atomic.AtomicInteger;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private AtomicInteger tokens;\n    private static final int MAX_TOKENS = 10;\n\n    private {name}() {{\n        tokens = new AtomicInteger(MAX_TOKENS);\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public boolean allowRequest() {{\n        if (tokens.get() > 0) {{\n            tokens.decrementAndGet();\n            return true;\n        }} else {{\n            System.out.println(\"Request denied: Rate limit exceeded\");\n            return false;\n        }}\n    }}\n\n    public void refillTokens() {{\n        tokens.set(MAX_TOKENS);\n    }}\n\n}}"
    },
    {
        "type": "GlobalIdGeneratorSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.concurrent.atomic.AtomicInteger;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private AtomicInteger counter;\n\n    private {name}() {{\n        counter = new AtomicInteger(0);\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public int generate() {{\n        return counter.incrementAndGet();\n    }}\n\n}}"
    },
    {
        "type": "DistributedLockManagerSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.concurrent.ConcurrentHashMap;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private ConcurrentHashMap<String, Boolean> locks;\n\n    private {name}() {{\n        locks = new ConcurrentHashMap<>();\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public boolean acquireLock(String key) {{\n        return locks.putIfAbsent(key, true) == null;\n    }}\n\n    public void releaseLock(String key) {{\n        locks.remove(key);\n    }}\n\n}}"
    },
    {
        "type": "RealTimeEventPublisherSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.concurrent.CopyOnWriteArrayList;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private CopyOnWriteArrayList<String> eventListeners;\n\n    private {name}() {{\n        eventListeners = new CopyOnWriteArrayList<>();\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void addListener(String listener) {{\n        eventListeners.add(listener);\n    }}\n\n    public void publishEvent(String event) {{\n        for (String listener : eventListeners) {{\n            System.out.println(\"Notifying listener: \" + listener + \" of event: \" + event);\n        }}\n    }}\n\n}}"
    },
    {
        "type": "SingletonWithMetricsCollection",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.ArrayList;\nimport java.util.List;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private List<String> collectedMetrics;\n\n    private {name}() {{\n        collectedMetrics = new ArrayList<>();\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void collectMetric(String metric) {{\n        collectedMetrics.add(metric);\n        System.out.println(\"Collected metric: \" + metric);\n    }}\n\n    public List<String> getCollectedMetrics() {{\n        return collectedMetrics;\n    }}\n\n}}"
    },
    {
        "type": "SingletonWithMetricsCollection",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.ArrayList;\nimport java.util.List;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private List<String> collectedMetrics;\n\n    private {name}() {{\n        collectedMetrics = new ArrayList<>();\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void collectMetric(String metric) {{\n        collectedMetrics.add(metric);\n        System.out.println(\"Collected metric: \" + metric);\n    }}\n\n    public List<String> getCollectedMetrics() {{\n        return collectedMetrics;\n    }}\n\n}}"
    },
    {
        "type": "SingletonWithMetricsCollection",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.ArrayList;\nimport java.util.List;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private List<String> collectedMetrics;\n\n    private {name}() {{\n        collectedMetrics = new ArrayList<>();\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void collectMetric(String metric) {{\n        collectedMetrics.add(metric);\n        System.out.println(\"Collected metric: \" + metric);\n    }}\n\n    public List<String> getCollectedMetrics() {{\n        return collectedMetrics;\n    }}\n\n}}"
    },
    {
        "type": "SingletonWithMetricsCollection",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.ArrayList;\nimport java.util.List;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private List<String> collectedMetrics;\n\n    private {name}() {{\n        collectedMetrics = new ArrayList<>();\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void collectMetric(String metric) {{\n        collectedMetrics.add(metric);\n        System.out.println(\"Collected metric: \" + metric);\n    }}\n\n    public List<String> getCollectedMetrics() {{\n        return collectedMetrics;\n    }}\n\n}}"
    },
    {
        "type": "SingletonWithMetricsCollection",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.ArrayList;\nimport java.util.List;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private List<String> collectedMetrics;\n\n    private {name}() {{\n        collectedMetrics = new ArrayList<>();\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void collectMetric(String metric) {{\n        collectedMetrics.add(metric);\n        System.out.println(\"Collected metric: \" + metric);\n    }}\n\n    public List<String> getCollectedMetrics() {{\n        return collectedMetrics;\n    }}\n\n}}"
    },
    {
        "type": "SingletonWithMetricsCollection",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.ArrayList;\n   private static {name} instance;\n    private List<String> collectedMetrics;\n\n    private {name}() {{\n        collectedMetrics = new ArrayList<>();\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void collectMetric(String metric) {{\n        collectedMetrics.add(metric);\n        System.out.println(\"Collected metric: \" + metric);\n    }}\n\n    public List<String> getCollectedMetrics() {{\n        return collectedMetrics;\n    }}\n\n}}"
    },
    {
        "type": "DatabasePoolSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.sql.Connection;\nimport java.sql.DriverManager;\nimport java.sql.SQLException;\nimport java.util.concurrent.ArrayBlockingQueue;\nimport java.util.concurrent.BlockingQueue;\n\npublic class {name} {{\n    private static {name} instance;\n    private BlockingQueue<Connection> connectionPool;\n\n    private {name}() {{\n        connectionPool = new ArrayBlockingQueue<>(5);\n        try {{\n            for (int i = 0; i < 5; i++) {{\n                connectionPool.add(DriverManager.getConnection(\"jdbc:mysql://localhost:3306/db\", \"user\", \"password\"));\n            }}\n        }} catch (SQLException e) {{\n            e.printStackTrace();\n        }}\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public Connection getConnection() {{\n        try {{\n            return connectionPool.take();\n        }} catch (InterruptedException e) {{\n            throw new RuntimeException(\"No connections available\");\n        }}\n    }}\n\n    public void releaseConnection(Connection connection) {{\n        connectionPool.offer(connection);\n    }}\n}}"
    },
    {
        "type": "LoggerSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.io.FileWriter;\nimport java.io.IOException;\n\npublic class {name} {{\n    private static {name} instance;\n    private FileWriter fileWriter;\n\n    private {name}() {{\n        try {{\n            fileWriter = new FileWriter(\"application.log\", true);\n        }} catch (IOException e) {{\n            e.printStackTrace();\n        }}\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            synchronized ({name}.class) {{\n                if (instance == null) {{\n                    instance = new {name}();\n                }}\n            }}\n        }}\n        return instance;\n    }}\n\n    public synchronized void log(String level, String message) {{\n        try {{\n            fileWriter.write(level + \": \" + message + \"\\n\");\n            fileWriter.flush();\n        }} catch (IOException e) {{\n            e.printStackTrace();\n        }}\n    }}\n}}"
    },
    {
        "type": "ConfigurationManager",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public enum {name} {{\n    INSTANCE;\n\n    private Map<String, String> config;\n\n    {name}() {{\n        config = new HashMap<>();\n        config.put(\"url\", \"http://example.com\");\n        config.put(\"timeout\", \"5000\");\n    }}\n\n    public String getConfig(String key) {{\n        return config.get(key);\n    }}\n\n    public void setConfig(String key, String value) {{\n        config.put(key, value);\n    }}\n}}"
    },
    {
        "type": "DoubleCheckedLockingSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n    private static volatile {name} instance;\n\n    private {name}() {{\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            synchronized ({name}.class) {{\n                if (instance == null) {{\n                    instance = new {name}();\n                }}\n            }}\n        }}\n        return instance;\n    }}\n}}"
    },
    {
        "type": "ThreadLocalSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n    private static final ThreadLocal<{name}> threadLocalInstance =\n            ThreadLocal.withInitial({name}::new);\n\n    private {name}() {{\n    }}\n\n    public static {name} getInstance() {{\n        return threadLocalInstance.get();\n    }}\n}}"
    },
    {
        "type": "ExpiringSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.time.Instant;\n\npublic class {name} {{\n    private static {name} instance;\n    private Instant creationTime;\n\n    private {name}() {{\n        creationTime = Instant.now();\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null || Instant.now().isAfter(instance.creationTime.plusSeconds(60))) {{\n            synchronized ({name}.class) {{\n                if (instance == null || Instant.now().isAfter(instance.creationTime.plusSeconds(60))) {{\n                    instance = new {name}();\n                }}\n            }}\n        }}\n        return instance;\n    }}\n\n    public String getCreationTime() {{\n        return creationTime.toString();\n    }}\n}}"
    },
    {
        "type": "CurrencyConverterSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.HashMap;\nimport java.util.Map;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private Map<String, Double> exchangeRates;\n\n    private {name}() {{\n        exchangeRates = new HashMap<>();\n        exchangeRates.put(\"USD_EUR\", 0.85);\n        exchangeRates.put(\"EUR_USD\", 1.18);\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            synchronized ({name}.class) {{\n                if (instance == null) {{\n                    instance = new {name}();\n                }}\n            }}\n        }}\n        return instance;\n    }}\n\n    public double convert(String from, String to, double amount) {{\n        String key = from + \"_\" + to;\n        return amount * exchangeRates.getOrDefault(key, 1.0);\n    }}\n\n}}"
    },
    {
        "type": "FileSystemCacheSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.HashSet;\nimport java.util.Set;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private Set<String> cachedFiles;\n\n    private {name}() {{\n        cachedFiles = new HashSet<>();\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void cacheFile(String path) {{\n        cachedFiles.add(path);\n        System.out.println(\"Caching file: \" + path);\n    }}\n\n    public boolean isFileCached(String path) {{\n        return cachedFiles.contains(path);\n    }}\n\n}}"
    },
    {
        "type": "NotificationServiceSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.HashSet;\nimport java.util.Set;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private Set<String> registeredUsers;\n\n    private {name}() {{\n        registeredUsers = new HashSet<>();\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void registerUser(String user) {{\n        registeredUsers.add(user);\n    }}\n\n    public void sendNotification(String user, String message) {{\n        if (registeredUsers.contains(user)) {{\n            System.out.println(\"Notification sent to \" + user + \": \" + message);\n        }} else {{\n            System.out.println(\"User not registered: \" + user);\n        }}\n    }}\n\n}}"
    },
    {
        "type": "SessionManagerSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.HashMap;\nimport java.util.Map;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private Map<String, String> sessions;\n\n    private {name}() {{\n        sessions = new HashMap<>();\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            synchronized ({name}.class) {{\n                if (instance == null) {{\n                    instance = new {name}();\n                }}\n            }}\n        }}\n        return instance;\n    }}\n\n    public void createSession(String user) {{\n        sessions.put(user, \"Active\");\n        System.out.println(\"Session created for: \" + user);\n    }}\n\n    public boolean isSessionActive(String user) {{\n        return sessions.containsKey(user);\n    }}\n\n}}"
    },
    {
        "type": "ApplicationSettingsSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.Properties;\nimport java.io.FileInputStream;\nimport java.io.IOException;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private Properties properties;\n\n    private {name}() {{\n        properties = new Properties();\n        try {{\n            properties.load(new FileInputStream(\"appsettings.properties\"));\n        }} catch (IOException e) {{\n            e.printStackTrace();\n        }}\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public String getSetting(String key) {{\n        return properties.getProperty(key);\n    }}\n\n}}"
    },
    {
        "type": "LoggerSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.io.FileWriter;\nimport java.io.IOException;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private FileWriter fileWriter;\n\n    private {name}() {{\n        try {{\n            fileWriter = new FileWriter(\"log.txt\", true);\n        }} catch (IOException e) {{\n            e.printStackTrace();\n        }}\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void log(String message) {{\n        try {{\n            fileWriter.write(message + \"\\n\");\n            fileWriter.flush();\n        }} catch (IOException e) {{\n            e.printStackTrace();\n        }}\n    }}\n\n}}"
    },
    {
        "type": "DatabaseConnectionPoolSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.Queue;\nimport java.util.LinkedList;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private Queue<String> connectionPool;\n\n    private {name}() {{\n        connectionPool = new LinkedList<>();\n        connectionPool.add(\"Connection1\");\n        connectionPool.add(\"Connection2\");\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public String getConnection() {{\n        return connectionPool.poll();\n    }}\n\n    public void releaseConnection(String connection) {{\n        connectionPool.add(connection);\n    }}\n\n}}"
    },
    {
        "type": "APIRequestRateLimiterSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.concurrent.atomic.AtomicInteger;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private AtomicInteger tokenCount;\n    private static final int MAX_TOKENS = 100;\n\n    private {name}() {{\n        tokenCount = new AtomicInteger(MAX_TOKENS);\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public boolean allowRequest() {{\n        if (tokenCount.get() > 0) {{\n            tokenCount.decrementAndGet();\n            return true;\n        }}\n        return false;\n    }}\n\n    public void refillTokens() {{\n        tokenCount.set(MAX_TOKENS);\n    }}\n\n}}"
    },
    {
        "type": "ConfigurationCacheSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.HashMap;\nimport java.util.Map;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private Map<String, String> cache;\n\n    private {name}() {{\n        cache = new HashMap<>();\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void put(String key, String value) {{\n        cache.put(key, value);\n    }}\n\n    public String get(String key) {{\n        return cache.get(key);\n    }}\n\n}}"
    },
    {
        "type": "TaskSchedulerSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.concurrent.Executors;\nimport java.util.concurrent.ScheduledExecutorService;\nimport java.util.concurrent.TimeUnit;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private ScheduledExecutorService scheduler;\n\n    private {name}() {{\n        scheduler = Executors.newScheduledThreadPool(1);\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void scheduleTask(Runnable task, long delay, TimeUnit unit) {{\n        scheduler.schedule(task, delay, unit);\n    }}\n\n}}"
    },
    {
        "type": "AuthenticationManagerSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.HashMap;\nimport java.util.Map;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private Map<String, String> userDatabase;\n\n    private {name}() {{\n        userDatabase = new HashMap<>();\n        userDatabase.put(\"admin\", \"password123\");\n        userDatabase.put(\"user\", \"pass\");\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public boolean authenticate(String user, String password) {{\n        return password.equals(userDatabase.get(user));\n    }}\n\n}}"
    },
    {
        "type": "EmailSenderSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n\n    private {name}() {{}}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void sendEmail(String recipient, String subject, String body) {{\n        System.out.println(\"Email sent to: \" + recipient + \" | Subject: \" + subject);\n    }}\n\n}}"
    },
    {
        "type": "ThreadLocalSessionManagerSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static final ThreadLocal<{name}> threadLocalInstance = ThreadLocal.withInitial({name}::new);\n\n    private {name}() {{}}\n\n    public static {name} getInstance() {{\n        return threadLocalInstance.get();\n    }}\n\n    public void createSession(String sessionId) {{\n        System.out.println(Thread.currentThread().getName() + \" - Session Created: \" + sessionId);\n    }}\n\n}}"
    },
    {
        "type": "SessionTrackerSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.HashSet;\nimport java.util.Set;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private Set<String> activeSessions;\n\n    private {name}() {{\n        activeSessions = new HashSet<>();\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void trackSession(String sessionId) {{\n        activeSessions.add(sessionId);\n        System.out.println(\"Session tracked: \" + sessionId);\n    }}\n\n    public boolean isSessionActive(String sessionId) {{\n        return activeSessions.contains(sessionId);\n    }}\n\n}}"
    },
    {
        "type": "NotificationCenterSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.ArrayList;\nimport java.util.List;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private List<String> notifications;\n\n    private {name}() {{\n        notifications = new ArrayList<>();\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void sendNotification(String message) {{\n        notifications.add(message);\n        System.out.println(\"Notification sent: \" + message);\n    }}\n\n    public List<String> getNotifications() {{\n        return notifications;\n    }}\n\n}}"
    },
    {
        "type": "ResourceAllocatorSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.LinkedList;\nimport java.util.Queue;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private Queue<String> resources;\n\n    private {name}() {{\n        resources = new LinkedList<>();\n        resources.add(\"Resource1\");\n        resources.add(\"Resource2\");\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public String allocateResource() {{\n        return resources.poll();\n    }}\n\n    public void releaseResource(String resourceName) {{\n        resources.add(resourceName);\n    }}\n\n}}"
    },
    {
        "type": "GlobalEventBusSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.ArrayList;\nimport java.util.List;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private List<String> events;\n\n    private {name}() {{\n        events = new ArrayList<>();\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void postEvent(String event) {{\n        events.add(event);\n        System.out.println(\"Event posted: \" + event);\n    }}\n\n    public List<String> getPostedEvents() {{\n        return events;\n    }}\n\n}}"
    },
    {
        "type": "FileStorageManagerSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.HashMap;\nimport java.util.Map;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private Map<String, byte[]> fileStorage;\n\n    private {name}() {{\n        fileStorage = new HashMap<>();\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void saveFile(String fileName, byte[] data) {{\n        fileStorage.put(fileName, data);\n        System.out.println(\"File saved: \" + fileName);\n    }}\n\n    public byte[] getFile(String fileName) {{\n        return fileStorage.get(fileName);\n    }}\n\n}}"
    },
    {
        "type": "LocalizationManagerSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.HashMap;\nimport java.util.Map;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private Map<String, String> translations;\n\n    private {name}() {{\n        translations = new HashMap<>();\n        translations.put(\"hello\", \"Hello\");\n        translations.put(\"goodbye\", \"Goodbye\");\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public String getTranslation(String key) {{\n        return translations.getOrDefault(key, \"Translation not found\");\n    }}\n\n}}"
    },
    {
        "type": "AnalyticsTrackerSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.ArrayList;\nimport java.util.List;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private List<String> trackedEvents;\n\n    private {name}() {{\n        trackedEvents = new ArrayList<>();\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void trackEvent(String eventName) {{\n        trackedEvents.add(eventName);\n        System.out.println(\"Event tracked: \" + eventName);\n    }}\n\n    public List<String> getTrackedEvents() {{\n        return trackedEvents;\n    }}\n\n}}"
    },
    {
        "type": "SchedulerWithCronJobsSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.Timer;\nimport java.util.TimerTask;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private Timer timer;\n\n    private {name}() {{\n        timer = new Timer(true);\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void addJob(String job, long delay) {{\n        timer.schedule(new TimerTask() {{\n            @Override\n            public void run() {{\n                System.out.println(\"Executing job: \" + job);\n            }}\n        }}, delay);\n    }}\n\n}}"
    },
    {
        "type": "GlobalConfigurationSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.Properties;\nimport java.io.FileInputStream;\nimport java.io.IOException;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private Properties properties;\n\n    private {name}() {{\n        properties = new Properties();\n        try {{\n            properties.load(new FileInputStream(\"global_config.properties\"));\n        }} catch (IOException e) {{\n            e.printStackTrace();\n        }}\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public String getSetting(String key) {{\n        return properties.getProperty(key);\n    }}\n\n}}"
    },
    {
        "type": "EventLoggerSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.io.FileWriter;\nimport java.io.IOException;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private FileWriter writer;\n\n    private {name}() {{\n        try {{\n            writer = new FileWriter(\"events.log\", true);\n        }} catch (IOException e) {{\n            e.printStackTrace();\n        }}\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void logEvent(String event) {{\n        try {{\n            writer.write(event + \"\\n\");\n            writer.flush();\n        }} catch (IOException e) {{\n            e.printStackTrace();\n        }}\n    }}\n\n}}"
    },
    {
        "type": "ImageCacheSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.HashMap;\nimport java.util.Map;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private Map<String, byte[]> cache;\n\n    private {name}() {{\n        cache = new HashMap<>();\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void addImage(String imageName, byte[] imageData) {{\n        cache.put(imageName, imageData);\n    }}\n\n    public byte[] getImage(String imageName) {{\n        return cache.get(imageName);\n    }}\n\n}}"
    },
    {
        "type": "AppConfigManagerSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.Properties;\nimport java.io.FileInputStream;\nimport java.io.IOException;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private Properties config;\n\n    private {name}() {{\n        config = new Properties();\n        try {{\n            config.load(new FileInputStream(\"app_config.properties\"));\n        }} catch (IOException e) {{\n            e.printStackTrace();\n        }}\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public String getConfig(String key) {{\n        return config.getProperty(key);\n    }}\n\n}}"
    },
    {
        "type": "DatabaseMigrationManagerSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.ArrayList;\nimport java.util.List;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private List<String> executedMigrations;\n\n    private {name}() {{\n        executedMigrations = new ArrayList<>();\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void migrateDatabase(String migrationScript) {{\n        if (!executedMigrations.contains(migrationScript)) {{\n            executedMigrations.add(migrationScript);\n            System.out.println(\"Migration executed: \" + migrationScript);\n        }} else {{\n            System.out.println(\"Migration already executed: \" + migrationScript);\n        }}\n    }}\n\n}}"
    },
    {
        "type": "TaskQueueManagerSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.concurrent.LinkedBlockingQueue;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private LinkedBlockingQueue<String> taskQueue;\n\n    private {name}() {{\n        taskQueue = new LinkedBlockingQueue<>();\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void addTask(String task) {{\n        try {{\n            taskQueue.put(task);\n            System.out.println(\"Task added: \" + task);\n        }} catch (InterruptedException e) {{\n            Thread.currentThread().interrupt();\n        }}\n    }}\n\n    public String getNextTask() {{\n        return taskQueue.poll();\n    }}\n\n}}"
    },
    {
        "type": "FeatureFlagManagerSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.HashMap;\nimport java.util.Map;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private Map<String, Boolean> featureFlags;\n\n    private {name}() {{\n        featureFlags = new HashMap<>();\n        featureFlags.put(\"DarkMode\", true);\n        featureFlags.put(\"BetaFeatures\", false);\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public boolean isFeatureEnabled(String feature) {{\n        return featureFlags.getOrDefault(feature, false);\n    }}\n\n}}"
    },
    {
        "type": "GlobalThemeManagerSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n    private String currentTheme;\n\n    private {name}() {{\n        currentTheme = \"Light\";\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public String getCurrentTheme() {{\n        return currentTheme;\n    }}\n\n    public void setTheme(String theme) {{\n        currentTheme = theme;\n        System.out.println(\"Theme updated to: \" + theme);\n    }}\n\n}}"
    },
    {
        "type": "SecurityManagerSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.HashMap;\nimport java.util.Map;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private Map<String, String> permissions;\n\n    private {name}() {{\n        permissions = new HashMap<>();\n        permissions.put(\"admin\", \"*:*\" );\n        permissions.put(\"user\", \"read:resources\");\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public boolean hasPermission(String user, String resource) {{\n        String allowedPermissions = permissions.get(user);\n        return allowedPermissions != null && (allowedPermissions.equals(\"*:*\") || allowedPermissions.contains(resource));\n    }}\n\n}}"
    },
    {
        "type": "DistributedCacheSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.concurrent.ConcurrentHashMap;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private ConcurrentHashMap<String, String> cache;\n\n    private {name}() {{\n        cache = new ConcurrentHashMap<>();\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void addCacheEntry(String key, String value) {{\n        cache.put(key, value);\n    }}\n\n    public String getCacheEntry(String key) {{\n        return cache.get(key);\n    }}\n\n}}"
    },
    {
        "type": "LazyLoadedResourceManagerSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.LinkedList;\nimport java.util.Queue;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private Queue<String> availableResources;\n\n    private {name}() {{\n        availableResources = new LinkedList<>();\n        availableResources.add(\"Resource1\");\n        availableResources.add(\"Resource2\");\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public String allocateResource() {{\n        return availableResources.poll();\n    }}\n\n    public void releaseResource(String resource) {{\n        availableResources.add(resource);\n    }}\n\n}}"
    },
    {
        "type": "ThreadSafeFeatureToggleSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.concurrent.ConcurrentHashMap;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private ConcurrentHashMap<String, Boolean> toggles;\n\n    private {name}() {{\n        toggles = new ConcurrentHashMap<>();\n        toggles.put(\"NewFeature\", true);\n        toggles.put(\"BetaMode\", false);\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public boolean isFeatureEnabled(String feature) {{\n        return toggles.getOrDefault(feature, false);\n    }}\n\n    public void setFeature(String feature, boolean isEnabled) {{\n        toggles.put(feature, isEnabled);\n    }}\n\n}}"
    },
    {
        "type": "ConnectionPoolManagerSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.LinkedList;\nimport java.util.Queue;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private Queue<String> connectionPool;\n\n    private {name}() {{\n        connectionPool = new LinkedList<>();\n        connectionPool.add(\"Connection1\");\n        connectionPool.add(\"Connection2\");\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public String getConnection() {{\n        return connectionPool.poll();\n    }}\n\n    public void releaseConnection(String connection) {{\n        connectionPool.add(connection);\n    }}\n\n}}"
    },
    {
        "type": "NotificationServiceSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.ArrayList;\nimport java.util.List;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private List<String> notifications;\n\n    private {name}() {{\n        notifications = new ArrayList<>();\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void sendNotification(String message) {{\n        notifications.add(message);\n        System.out.println(\"Notification sent: \" + message);\n    }}\n\n    public List<String> getNotifications() {{\n        return notifications;\n    }}\n\n}}"
    },
    {
        "type": "EventBusSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.concurrent.CopyOnWriteArrayList;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private CopyOnWriteArrayList<String> subscribers;\n\n    private {name}() {{\n        subscribers = new CopyOnWriteArrayList<>();\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void subscribe(String subscriber) {{\n        subscribers.add(subscriber);\n    }}\n\n    public void postEvent(String event) {{\n        for (String subscriber : subscribers) {{\n            System.out.println(\"Event sent to \" + subscriber + \": \" + event);\n        }}\n    }}\n\n}}"
    },
    {
        "type": "ThreadLocalCounterSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static final ThreadLocal<{name}> threadLocalInstance = ThreadLocal.withInitial({name}::new);\n    private int counter;\n\n    private {name}() {{\n        counter = 0;\n    }}\n\n    public static {name} getInstance() {{\n        return threadLocalInstance.get();\n    }}\n\n    public int incrementAndGet() {{\n        return ++counter;\n    }}\n\n}}"
    },
    {
        "type": "GlobalRateLimiterSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.concurrent.ConcurrentHashMap;\nimport java.util.concurrent.atomic.AtomicInteger;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private ConcurrentHashMap<String, AtomicInteger> clientRequests;\n    private static final int MAX_REQUESTS = 5;\n\n    private {name}() {{\n        clientRequests = new ConcurrentHashMap<>();\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public boolean allowRequest(String clientId) {{\n        clientRequests.putIfAbsent(clientId, new AtomicInteger(0));\n        AtomicInteger requestCount = clientRequests.get(clientId);\n        if (requestCount.get() < MAX_REQUESTS) {{\n            requestCount.incrementAndGet();\n            return true;\n        }} else {{\n            System.out.println(\"Rate limit exceeded for: \" + clientId);\n            return false;\n        }}\n    }}\n\n}}"
    },
    {
        "type": "SingletonWithTimeoutInitialization",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.concurrent.locks.Lock;\nimport java.util.concurrent.locks.ReentrantLock;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private static final Lock lock = new ReentrantLock();\n\n    private {name}() {{}}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            try {{\n                if (lock.tryLock()) {{\n                    try {{\n                        if (instance == null) {{\n                            instance = new {name}();\n                        }}\n                    }} finally {{\n                        lock.unlock();\n                    }}\n                }}\n            }} catch (Exception e) {{\n                System.out.println(\"Timeout during initialization\");\n            }}\n        }}\n        return instance;\n    }}\n\n    public void initialize() {{\n        System.out.println(\"Initialized\");\n    }}\n\n}}"
    },
    {
        "type": "SingletonWithCyclicDependencyGuard",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n    private static boolean isCreating;\n\n    private {name}() {{\n        if (isCreating) {{\n            throw new IllegalStateException(\"Cyclic dependency detected!\");\n        }}\n        isCreating = true;\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n            isCreating = false;\n        }}\n        return instance;\n    }}\n\n}}"
    },
    {
        "type": "SingletonWithFallbackInstance",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n    private static final {name} fallbackInstance = new {name}(\"fallback\");\n\n    private String mode;\n\n    private {name}(String mode) {{\n        this.mode = mode;\n    }}\n\n    public static {name} getInstance(boolean useFallback) {{\n        if (useFallback) {{\n            return fallbackInstance;\n        }}\n        if (instance == null) {{\n            instance = new {name}(\"primary\");\n        }}\n        return instance;\n    }}\n\n    public String getMode() {{\n        return mode;\n    }}\n\n}}"
    },
    {
        "type": "SelfDestructingSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n    private static long lastAccessed;\n    private static final long TIMEOUT = 3000; // milliseconds\n\n    private {name}() {{}}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null || (System.currentTimeMillis() - lastAccessed) > TIMEOUT) {{\n            instance = new {name}();\n        }}\n        lastAccessed = System.currentTimeMillis();\n        return instance;\n    }}\n\n}}"
    },
    {
        "type": "DistributedLockingSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.concurrent.ConcurrentHashMap;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private ConcurrentHashMap<String, Boolean> lockMap;\n\n    private {name}() {{\n        lockMap = new ConcurrentHashMap<>();\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public boolean acquireLock(String key) {{\n        return lockMap.putIfAbsent(key, true) == null;\n    }}\n\n    public void releaseLock(String key) {{\n        lockMap.remove(key);\n    }}\n\n}}"
    },
    {
        "type": "SingletonWithObserverLifecycle",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.ArrayList;\nimport java.util.List;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private List<String> events;\n\n    private {name}() {{\n        events = new ArrayList<>();\n        onStart();\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void onStart() {{\n        System.out.println(\"Singleton started\");\n    }}\n\n    public void onEvent(String event) {{\n        events.add(event);\n        System.out.println(\"Event captured: \" + event);\n    }}\n\n    public void onStop() {{\n        events.clear();\n        System.out.println(\"Singleton stopped\");\n    }}\n\n}}"
    },
    {
        "type": "SynchronizedMethodSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n\n    private {name}() {{}}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n}}"
    },
    {
        "type": "DoubleCheckedLockingSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static volatile {name} instance;\n\n    private {name}() {{}}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            synchronized ({name}.class) {{\n                if (instance == null) {{\n                    instance = new {name}();\n                }}\n            }}\n        }}\n        return instance;\n    }}\n\n}}"
    },
    {
        "type": "BillPughSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private {name}() {{}}\n\n    private static class SingletonHelper {{\n        private static final {name} INSTANCE = new {name}();\n    }}\n\n    public static {name} getInstance() {{\n        return SingletonHelper.INSTANCE;\n    }}\n\n}}"
    },
    {
        "type": "ThreadLocalSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static final ThreadLocal<{name}> instance = ThreadLocal.withInitial({name}::new);\n\n    private {name}() {{}}\n\n    public static {name} getInstance() {{\n        return instance.get();\n    }}\n\n}}"
    },
    {
        "type": "EnumBasedSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public enum {name} {{\n\n    INSTANCE;\n\n    public void performAction() {{\n        System.out.println(\"Action performed by Singleton\");\n    }}\n\n}}"
    },
    {
        "type": "StaticBlockInitializationSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n\n    static {{\n        try {{\n            instance = new {name}();\n        }} catch (Exception e) {{\n            throw new RuntimeException(\"Error initializing singleton\", e);\n        }}\n    }}\n\n    private {name}() {{}}\n\n    public static {name} getInstance() {{\n        return instance;\n    }}\n\n}}"
    },
    {
        "type": "VolatileLazyInitializationSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static volatile {name} instance;\n\n    private {name}() {{}}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            synchronized ({name}.class) {{\n                if (instance == null) {{\n                    instance = new {name}();\n                }}\n            }}\n        }}\n        return instance;\n    }}\n\n}}"
    },
    {
        "type": "ThreadSafeLazyHolderSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private {name}() {{}}\n\n    private static class LazyHolder {{\n        private static final {name} INSTANCE = new {name}();\n    }}\n\n    public static {name} getInstance() {{\n        return LazyHolder.INSTANCE;\n    }}\n\n}}"
    },
    {
        "type": "SynchronizedBlockSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n\n    private {name}() {{}}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            synchronized ({name}.class) {{\n                if (instance == null) {{\n                    instance = new {name}();\n                }}\n            }}\n        }}\n        return instance;\n    }}\n\n}}"
    },
    {
        "type": "ThreadSafeInitializationOnDemandHolder",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private {name}() {{}}\n\n    private static class Holder {{\n        private static final {name} INSTANCE = new {name}();\n    }}\n\n    public static {name} getInstance() {{\n        return Holder.INSTANCE;\n    }}\n\n}}"
    },
    {
        "type": "EagerThreadSafeSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static final {name} INSTANCE = new {name}();\n\n    private {name}() {{}}\n\n    public static {name} getInstance() {{\n        return INSTANCE;\n    }}\n\n}}"
    },
    {
        "type": "SynchronizedDoubleCheckSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static volatile {name} instance;\n\n    private {name}() {{}}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            synchronized ({name}.class) {{\n                if (instance == null) {{\n                    instance = new {name}();\n                }}\n            }}\n        }}\n        return instance;\n    }}\n\n}}"
    },
    {
        "type": "SynchronizedAccessSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n\n    private {name}() {{}}\n\n    public static {name} getInstance() {{\n        synchronized ({name}.class) {{\n            if (instance == null) {{\n                instance = new {name}();\n            }}\n        }}\n        return instance;\n    }}\n\n}}"
    },
    {
        "type": "ConcurrentMapSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.concurrent.ConcurrentHashMap;\n\npublic class {name} {{\n\n    private static final ConcurrentHashMap<String, {name}> instances = new ConcurrentHashMap<>();\n\n    private {name}() {{}}\n\n    public static {name} getInstance(String key) {{\n        return instances.computeIfAbsent(key, k -> new {name}());\n    }}\n\n}}"
    },
    {
        "type": "AtomicReferenceSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.concurrent.atomic.AtomicReference;\n\npublic class {name} {{\n\n    private static final AtomicReference<{name}> INSTANCE = new AtomicReference<>();\n\n    private {name}() {{}}\n\n    public static {name} getInstance() {{\n        if (INSTANCE.get() == null) {{\n            INSTANCE.compareAndSet(null, new {name}());\n        }}\n        return INSTANCE.get();\n    }}\n\n}}"
    },
    {
        "type": "ThreadSafeSingletonWithCounter",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n    private int counter;\n\n    private {name}() {{\n        counter = 0;\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public synchronized void incrementCounter() {{\n        counter++;\n    }}\n\n    public synchronized int getCounter() {{\n        return counter;\n    }}\n\n}}"
    },
    {
        "type": "ThreadSafeSingletonWithThreadPool",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.concurrent.ExecutorService;\nimport java.util.concurrent.Executors;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private ExecutorService executorService;\n\n    private {name}() {{\n        executorService = Executors.newFixedThreadPool(5);\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public ExecutorService getExecutorService() {{\n        return executorService;\n    }}\n\n}}"
    },
    {
        "type": "ThreadSafeSingletonWithCache",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.concurrent.ConcurrentHashMap;\nimport java.util.Map;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private Map<String, String> cache;\n\n    private {name}() {{\n        cache = new ConcurrentHashMap<>();\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public synchronized void addToCache(String key, String value) {{\n        cache.put(key, value);\n    }}\n\n    public synchronized String getFromCache(String key) {{\n        return cache.get(key);\n    }}\n\n}}"
    },
    {
        "type": "ThreadSafeSingletonWithLifecycle",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n\n    private {name}() {{\n        onCreate();\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    private void onCreate() {{\n        System.out.println(\"Singleton created\");\n    }}\n\n    public void onDestroy() {{\n        System.out.println(\"Singleton destroyed\");\n        instance = null;\n    }}\n\n}}"
    },
    {
        "type": "SynchronizedWeakReferenceSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.lang.ref.WeakReference;\n\npublic class {name} {{\n\n    private static WeakReference<{name}> instance;\n\n    private {name}() {{}}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null || instance.get() == null) {{\n            instance = new WeakReference<>(new {name}());\n        }}\n        return instance.get();\n    }}\n\n}}"
    },
    {
        "type": "SingletonWithRetryPolicy",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n    private int retryCount;\n\n    private {name}() {{\n        retryCount = 0;\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void retry(String task) {{\n        retryCount++;\n        System.out.println(\"Retrying task: \" + task + \" - Retry Count: \" + retryCount);\n    }}\n\n    public int getRetryCount() {{\n        return retryCount;\n    }}\n\n}}"
    },
    {
        "type": "LazySingletonWithInitializationCheck",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n\n    private {name}() {{}}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public boolean isInitialized() {{\n        return instance != null;\n    }}\n\n}}"
    },
    {
        "type": "SingletonWithSessionTimeout",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.HashMap;\nimport java.util.Map;\nimport java.util.concurrent.Executors;\nimport java.util.concurrent.ScheduledExecutorService;\nimport java.util.concurrent.TimeUnit;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private Map<String, Long> sessions;\n    private ScheduledExecutorService scheduler;\n\n    private {name}() {{\n        sessions = new HashMap<>();\n        scheduler = Executors.newScheduledThreadPool(1);\n        scheduler.scheduleAtFixedRate(() -> {{\n            long currentTime = System.currentTimeMillis();\n            sessions.entrySet().removeIf(entry -> currentTime - entry.getValue() > 30000);\n        }}, 0, 10, TimeUnit.SECONDS);\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void createSession(String user) {{\n        sessions.put(user, System.currentTimeMillis());\n        System.out.println(\"Session created for: \" + user);\n    }}\n\n    public boolean isSessionActive(String user) {{\n        return sessions.containsKey(user);\n    }}\n\n}}"
    },
    {
        "type": "SingletonWithDynamicResourcePool",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.concurrent.ConcurrentLinkedQueue;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private ConcurrentLinkedQueue<String> resources;\n\n    private {name}() {{\n        resources = new ConcurrentLinkedQueue<>();\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void addResource(String resource) {{\n        resources.add(resource);\n    }}\n\n    public String getResource() {{\n        return resources.poll();\n    }}\n\n}}"
    },
    {
        "type": "DistributedSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.concurrent.ConcurrentHashMap;\n\npublic class {name} {{\n\n    private static ConcurrentHashMap<String, {name}> instances = new ConcurrentHashMap<>();\n\n    private {name}() {{}}\n\n    public static {name} getInstance(String region) {{\n        return instances.computeIfAbsent(region, k -> new {name}());\n    }}\n\n}}"
    },
    {
        "type": "SingletonWithPreloadedData",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.HashMap;\nimport java.util.Map;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private Map<String, String> data;\n\n    private {name}() {{\n        data = new HashMap<>();\n        data.put(\"Key1\", \"Value1\");\n        data.put(\"Key2\", \"Value2\");\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public String getData(String key) {{\n        return data.get(key);\n    }}\n\n}}"
    },
    {
        "type": "SingletonWithRuntimeExceptionProtection",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n\n    private {name}() {{\n        if (instance != null) {{\n            throw new RuntimeException(\"Instance already created\");\n        }}\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n}}"
    },
    {
        "type": "SingletonWithInitializationHooks",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n\n    private {name}() {{\n        initialize();\n    }}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    private void initialize() {{\n        System.out.println(\"Singleton Initialized\");\n    }}\n\n}}"
    },
    {
        "type": "SingletonWithOptionalInstance",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.Optional;\n\npublic class {name} {{\n\n    private static {name} instance;\n\n    private {name}() {{}}\n\n    public static Optional<{name}> getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return Optional.of(instance);\n    }}\n\n}}"
    },
    {
        "type": "ParameterizedLazySingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n    private String config;\n\n    private {name}(String config) {{\n        this.config = config;\n    }}\n\n    public static {name} getInstance(String config) {{\n        if (instance == null) {{\n            instance = new {name}(config);\n        }}\n        return instance;\n    }}\n\n    public String getConfig() {{\n        return config;\n    }}\n\n}}"
    },
    {
        "type": "ThreadSafeLazySingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static volatile {name} instance;\n\n    private {name}() {{}}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            synchronized ({name}.class) {{\n                if (instance == null) {{\n                    instance = new {name}();\n                }}\n            }}\n        }}\n        return instance;\n    }}\n\n}}"
    },
    {
        "type": "SingletonWithDynamicConfiguration",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n    private String environment;\n\n    private {name}(String environment) {{\n        this.environment = environment;\n    }}\n\n    public static {name} getInstance(String environment) {{\n        if (instance == null) {{\n            instance = new {name}(environment);\n        }}\n        return instance;\n    }}\n\n    public String getEnvironment() {{\n        return environment;\n    }}\n\n}}"
    },
    {
        "type": "SingletonWithMultipleConstructors",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n    private String config;\n\n    private {name}(String config) {{\n        this.config = config;\n    }}\n\n    private {name}() {{}}\n\n    public static {name} getInstance(String config) {{\n        if (instance == null) {{\n            instance = new {name}(config);\n        }}\n        return instance;\n    }}\n\n    public String getConfig() {{\n        return config;\n    }}\n\n}}"
    },
    {
        "type": "SingletonWithInheritance",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}\n\npublic class DerivedSingleton extends {name} {{\n\n    private static DerivedSingleton instance;\n\n    private DerivedSingleton() {{}}\n\n    public static DerivedSingleton getInstance() {{\n        if (instance == null) {{\n            instance = new DerivedSingleton();\n        }}\n        return instance;\n    }}\n\n}}"
    },
    {
        "type": "SingletonWithInitializationLogging",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n\n    private {name}() {{\n        System.out.println(\"{name} initialized\");\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n}}"
    },
    {
        "type": "SingletonWithImmutableState",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n    private final String state;\n\n    private {name}(String state) {{\n        this.state = state;\n    }}\n\n    public static {name} getInstance(String state) {{\n        if (instance == null) {{\n            instance = new {name}(state);\n        }}\n        return instance;\n    }}\n\n    public String getState() {{\n        return state;\n    }}\n\n}}"
    },
    {
        "type": "ThreadLocalSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static final ThreadLocal<{name}> threadLocalInstance = ThreadLocal.withInitial({name}::new);\n\n    private {name}() {{}}\n\n    public static {name} getInstance() {{\n        return threadLocalInstance.get();\n    }}\n\n}}"
    },
    {
        "type": "SingletonWithFallbackInstance",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n\n    private {name}() {{}}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            try {{\n                instance = new {name}();\n            }} catch (Exception e) {{\n                instance = new {name}(); // Fallback logic\n            }}\n        }}\n        return instance;\n    }}\n\n}}"
    }
]

# Generate examples for each pattern
examples = []
for pattern in patterns:
    for i in range(5):  # Generate 5 variations per pattern
        class_name = f"{pattern['type']}Example{i + 1}"
        examples.append({
            "type": pattern["type"],
            "input": pattern["input_template"].format(name=class_name),
            "output": pattern["output_template"].format(name=class_name)
        })

# Save the dataset to a JSON file
dataset_filename = "augmented_singleton_data.json"
with open(dataset_filename, "w") as json_file:
    json.dump(examples, json_file, indent=4)

print(f"Dataset generated and saved to {dataset_filename}.")
