using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Net.Sockets;
using Debug = UnityEngine.Debug;
using System.Windows.Input;

public class UserComponent : MonoBehaviour
{
    // Debugging variables
    public static bool PRINT_TO_CONSOLE = false;
    public static bool PRINT_CRITICAL = true;
    public static bool MANUAL_CONTROL = false;
    public static float[] expectedAngles;

    // Game variables
    private UnityEngine.GameObject[] movableLimbs;
    private UnityEngine.Rigidbody[] rigidBodies;
    private float[] startingAngles; //  in radians
    private Vector3[] startingPositions;
    private float[] torquesToApply;
    private float[] limbData; // in radians
    private object limbData_locker = new object();
    private object torquesToApply_locker = new object();

    // Process/training variables
    private bool training = true; // To be hard-coded (for now)
    private ClientConnectionHandler connection;
    private Process process = null;
    private Thread thread = null;
    private bool running = true;
    private bool requestReset = false;
    private Stopwatch stopwatch = null;

    private bool zeroPosition = true;

    private int positionSetupResets = 300;
    // private bool requestLimbData = false;
    // private string stringLimbData = "";

    // Hands-on training variables
    private int resetCount = 0;
    // private long nextFrameTimeMs = 0;
    // private bool waitingForNewFrame;

    // Start is called before the first frame update
    void Start()
    {
        // Object fetching related logic
        while (GeneralData.isReady() == false)
        {
        }

        movableLimbs = GeneralData.getHingeLimbs("Hand");
        rigidBodies = new Rigidbody[movableLimbs.Length];

        for (int i = 0; i < movableLimbs.Length; i++)
        {
            rigidBodies[i] = ((GameObject) movableLimbs[i]).GetComponent(typeof(Rigidbody)) as Rigidbody;
            rigidBodies[i].maxAngularVelocity = 1000;
        }

        startingAngles = new float[movableLimbs.Length];
        startingPositions = new Vector3[movableLimbs.Length];

        torquesToApply = new float[movableLimbs.Length];
        limbData = new float[movableLimbs.Length * 2];

        // Thread instantiation
        if (training == true)
        {
            thread = new Thread(this.RunTrainingThread);
            thread.Start();
        }

        print("Finished the basic initializations of the program.");

        // Initializations for the frame viewer
        expectedAngles = new float[movableLimbs.Length];
    }

    private void RunTrainingThread()
    {
        /*
         * Starting Sequence
         */

        // Model related logic
        connection = new ClientConnectionHandler();
        process = new Process();

        // Calls python training script.
        process.StartInfo.FileName = @"C:\Users\Michael\AppData\Local\Microsoft\WindowsApps\python.exe";
        string scriptPath = @"C:\Git\Virtual-Hand\PythonScripts\ModelTrainerV4.py";
        string dataSetName = "RealData15_smoothed"; // To be hard-coded (for now)
        string modelName = "FirstModelTest";
        process.StartInfo.Arguments = scriptPath;


        // Starts the process
        print("Starting the process: " + process.StartInfo.FileName);
        process.StartInfo.UseShellExecute = false;
        process.StartInfo.CreateNoWindow = true;
        process.Start();
        System.Threading.Thread.Sleep(6000);

        connection.println(dataSetName);
        connection.println(modelName);

        print("Started the Python process. ");

        // Interactions with the Python Script
        // string acknowledgement = stdoutReadLine();
        string acknowledgement = connection.readline();
        print("Acknowledgement from Python: " + acknowledgement);

        if (acknowledgement.Equals("Ready") == false)
        {
            print("Did not receive acknowledgement from Python script.");
            Quit();
        }
        else
        {
            connection.println("Ready");

            // Obtains starting angles from the python script
            print("Reading start angles...");
            // string[] stringBaseAngles = connection.readline().Split(' ');
            // for (int i = 0; i < stringBaseAngles.Length; i++)
            // {
            //     print(stringBaseAngles[i]);
            //     startingAngles[i] = float.Parse(stringBaseAngles[i]);
            // }

            startingAngles = GeneralData.string2floatArray(connection.readline());

            // startingAngles = new float[movableLimbs.Length]; // todo, remove this

            print("Expecting start angles...");
            foreach (var angle in startingAngles)
            {
                controlled_print(angle.ToString());
            }

            controlled_print("Python angles obtained: " + startingAngles.ToString());

            // waitingForNewFrame = true;
            ResetTrainingSequence_forThread(true);
            Ready();
        }

        // Resets the system multiple times to achieve stable start position
        while (positionSetupResets > 0)
        {
            if (positionSetupResets % 100 == 0)
            {
                critical_print("Positional resets remaining: " + positionSetupResets.ToString());
            }

            zeroPosition = true;
            ResetTrainingSequence_forThread(false);
            positionSetupResets = positionSetupResets - 1;
        }

        // Initial ready (other ready's are received upon restart)
        if (connection.readline().Equals("Ready") == false)
        {
            critical_print("False start! (did not receive loop ready)");
            Quit();
        }

        /*
         * Process Loop
         */
        while (running == true)
        {
            // Step Loop-0 (as per Pprotocol)
            // controlled_print("Stopwatch: " + stopwatch.ElapsedMilliseconds.ToString());
            // if (waitingForNewFrame == true)
            // {
            // controlled_print("Next will be: ");
            // nextFrameTimeMs = long.Parse(connection.readline());
            // controlled_print("Next time frame: " + nextFrameTimeMs);
            // waitingForNewFrame = false;
            // }
            // else if (stopwatch.ElapsedMilliseconds >= nextFrameTimeMs) // Step Loop-1 (as per protocol)
            // {
            // Step Loop-2 (as per protocol)
            string toSend = GetStringLimbData_forThread();

            // Sends data to python script
            // Step Loop-3 (as per protocol)
            long current_time_ms = stopwatch.ElapsedMilliseconds;
            connection.println(current_time_ms.ToString());

            critical_print("Current time: " + current_time_ms.ToString());

            // Step Loop-4 (as per protocol)
            connection.println(toSend);

            expectedAngles =
                GeneralData.string2floatArray(connection
                    .readline()); // Obtains the expected angles from the Python Script

            // Step Loop-5 (as per protocol)
            string nextCommand = connection.readline();
            // Step Loop-6 (as per protocol)
            if (nextCommand.Equals("Reset"))
            {
                if (FrameViewer.earlyReset == true)
                {
                    ResetTrainingSequence_forThread(true);
                }

                if (connection.readline().Equals("Ready"))
                    ResetTrainingSequence_forThread(true);
                else
                {
                    critical_print("We've got a big mistake. No reset Ready received.");
                }

                Ready();
            }
            else if (nextCommand.Equals("Quit"))
            {
                Quit();
            }
            else if (nextCommand.Equals("Next"))
            {
                // Obtains and applies torques from python script to the limbs
                string stringTorques = connection.readline();
                controlled_print("Received Torques length: " + stringTorques.ToString());

                lock (torquesToApply_locker)
                {
                    // for (int i = 0; i < movableLimbs.Length; i++)
                    // {
                    //     torquesToApply[i] = float.Parse(stringTorques[i]);
                    // }

                    torquesToApply = GeneralData.string2floatArray(stringTorques);
                }

                // waitingForNewFrame = true;
                Ready();
            }
            else
            {
                print(
                    "Unknown nextCommand sent from python script (" + nextCommand + "). Aborting program.");
                Quit();
            }

            // }
        }
    }

    /*
     * FixedUpdates refreshes 50 times per second by default
     */
    private void FixedUpdate()
    {
        // Continuously applies existing torques to the limbs
        lock (torquesToApply_locker)
        {
            for (int i = 0; i < movableLimbs.Length; i++)
            {
                // if(i == 4){
                // rigidBodies[i].AddTorque(new Vector3(torquesToApply[i], 0, 0), ForceMode.Force); // todo, old command, maybe change back?
                // rigidBodies[i].AddTorque(new Vector3(torquesToApply[i], 0, 0), ForceMode.VelocityChange); // todo, old command, maybe change back?

                // if (i != 4)
                // {
                // torquesToApply[i] = 0;
                // }

                if (i % 3 != 0
                ) // TODO, if this works, then make a function that searches for the nearest parent with a rigid body (if exists)
                {
                    rigidBodies[i].angularVelocity =
                        new Vector3(
                            movableLimbs[i].transform.parent.GetComponent<Rigidbody>().angularVelocity.x +
                            torquesToApply[i], 0, 0);
                }
                else
                {
                    rigidBodies[i].angularVelocity = new Vector3(torquesToApply[i], 0, 0);
                }
                // }
                // else // todo, don't forget to remove this
                // {
                // movableLimbs[i].transform.localRotation = Quaternion.Euler(expectedAngles[i] * 57.29577951f, 0, 0);
                // rigidBodies[i] = new Vector3(0, 0, 0);
                // movableLimbs[i].transform.lo

                // }
            }
        }

        if (MANUAL_CONTROL == true)
        {
            float force = 0;
            if (Input.GetKey(KeyCode.Q))
            {
                force += 0.0000000000000001f;
            }

            if (Input.GetKey(KeyCode.W))
            {
                force += 0.000000000000001f;
            }

            if (Input.GetKey(KeyCode.E))
            {
                force += 0.00000000000001f;
            }

            if (Input.GetKey(KeyCode.R))
            {
                force += 0.0000000000001f;
            }

            if (Input.GetKey(KeyCode.T))
            {
                force += 0.000000000001f;
            }

            if (Input.GetKey(KeyCode.Y))
            {
                force += 0.00000000001f;
            }

            if (Input.GetKey(KeyCode.U))
            {
                force += 0.0000000001f;
            }

            if (Input.GetKey(KeyCode.I))
            {
                force += 0.000000001f;
            }

            if (Input.GetKey(KeyCode.O))
            {
                force += 0.00000001f;
            }

            if (Input.GetKey(KeyCode.P))
            {
                force += 0.0000001f;
            }

            if (Input.GetKey(KeyCode.L))
            {
                force += 0.000001f;
            }

            if (Input.GetKey(KeyCode.K))
            {
                force += 0.00001f;
            }

            if (Input.GetKey(KeyCode.J))
            {
                force += 0.0001f;
            }

            if (Input.GetKey(KeyCode.H))
            {
                force += 0.001f;
            }

            if (Input.GetKey(KeyCode.H))
            {
                force += 0.01f;
            }

            if (Input.GetKey(KeyCode.G))
            {
                force += 0.1f;
            }

            if (Input.GetKey(KeyCode.F))
            {
                force += 1.0f;
            }

            if (Input.GetKey(KeyCode.D))
            {
                force += 10.0f;
            }

            if (Input.GetKey(KeyCode.S))
            {
                force += 1000.0f;
            }

            if (Input.GetKey(KeyCode.A))
            {
                force += 10000.0f;
            }

            if (Input.GetKey(KeyCode.Z))
            {
                force = -force;
            }

            if (force > 0.000000000001f)
            {
                print("Yes, Force.");
            }

            if (Input.GetKey(KeyCode.X))
            {
                print("AAAAAAAAAAA");
            }

            for (int i = 0; i < movableLimbs.Length; i++)
            {
                rigidBodies[i].AddTorque(new Vector3(force, 0, 0), ForceMode.Force);
            }
        }
    }

    /*
     * Refreshes as fast as the frame rate.
     */
    private void Update()
    {
        if (requestReset == true)
        {
            ResetTrainingSequence();
        }

        for (int i = 0; i < movableLimbs.Length; i++)
        {
            lock (limbData_locker)
            {
                limbData[i * 2] = movableLimbs[i].transform.localEulerAngles.x;
                limbData[i * 2 + 1] = rigidBodies[i].angularVelocity.x;
            }
        }
    }

    /*
     * ResetTrainingSequence
     * Organizes the reset of the unity hand between the main and logic thread.
     */
    private void ResetTrainingSequence_forThread(bool countReset)
    {
        if (countReset == true)
        {
            resetCount++;
            critical_print("The system is resetting. Reset #" + resetCount);
        }

        requestReset = true;
        while (requestReset != false) // Waits until the main thread resets the hand
        {
            System.Threading.Thread.Sleep(1); // Sleeps for 1 ms while waiting
        }

        // Resets the "start" of unity's training sequence
        if (stopwatch != null)
        {
            stopwatch.Stop();
        }

        stopwatch = new Stopwatch();
        stopwatch.Start();
    }

    private void ResetTrainingSequence()
    {
        // Resets the limb positions
        for (int i = 0; i < movableLimbs.Length; i++)
        {
            // movableLimbs[i].transform.eulerAngles.Set(startingAngles[i], 0, 0);
            // movableLimbs[i].transform.localRotation = Quaternion.Euler(startingAngles[i], 0f, 0f);
            // movableLimbs[i].transform.SetPositionAndRotation(startingPositions[i],
            // Quaternion.Euler(startingAngles[i] * 57.29577958f, 0f, 0f));
            movableLimbs[i].transform.localRotation = Quaternion.Euler(startingAngles[i] * 57.29577951f, 0, 0);
            if (zeroPosition == false)
            {
                movableLimbs[i].transform.localPosition = startingPositions[i];
            }

            torquesToApply[i] = 0;
            rigidBodies[i].velocity = Vector3.zero;
            rigidBodies[i].angularVelocity = Vector3.zero;
            rigidBodies[i].Sleep();
        }

        // Obtains starting position if it does not exist
        if (zeroPosition == true)
        {
            for (int i = 0; i < movableLimbs.Length; i++)
            {
                startingPositions[i] = movableLimbs[i].transform.localPosition;
            }

            zeroPosition = false;
        }

        requestReset = false;
    }

    /*
     * GetStringTorques
     * Organizes the retrieval of the unity hand angles between the main and logic thread.
     */
    private string GetStringLimbData_forThread()
    {
        string stringLimbData = "";
        lock (limbData_locker)
        {
            for (int i = 0; i < limbData.Length; i++)
            {
                float angle = limbData[i];
                for (int j = 0; j < 10; j++) // todo, replace this with something more reliable
                {
                    if (angle > 180)
                    {
                        angle -= 360;
                    }
                    else if (angle < -180)
                    {
                        angle += 360;
                    }
                    else
                    {
                        break;
                    }
                }

                stringLimbData += (angle * 0.01745329252f).ToString() + " ";
            }
        }

        stringLimbData = stringLimbData.TrimEnd(' ');
        return stringLimbData;
    }

    /*
     * getMilisecond
     * Returns the universal time in miliseconds.
     */
    private void Ready()
    {
        connection.println("Ready");
    }

    public static void controlled_print(string message)
    {
        if (PRINT_TO_CONSOLE == true)
        {
            Debug.Log(message);
        }
    }

    public static void critical_print(string message)
    {
        if (PRINT_CRITICAL == true)
        {
            Debug.Log("Critical: " + message);
        }
        else
        {
            controlled_print(message);
        }
    }

    private void Quit()
    {
        running = false; // Stops the thread loop
        process.Close();
        connection.stop();
        print("Stopped.");
    }

    private void OnDestroy()
    {
        Quit();
    }
}