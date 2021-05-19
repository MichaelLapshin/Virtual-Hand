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

public class HandController : MonoBehaviour
{
    // Game variables
    private UnityEngine.GameObject[] movableLimbs;
    private UnityEngine.Rigidbody[] rigidBodies;
    private float[] limb_velocities; // velocity that should be applied in radians
    private float[] limbData; // angular. Position in degrees, velocity in radians

    // Process variables
    private ClientConnectionHandler connection;
    private Process process = null;
    private Thread thread = null;
    private bool running = false;
    private object limb_velocities_locker = new object();

    // Start is called before the first frame update
    void Start()
    {
        // Object fetching related logic
        print("Started the hand controller.");
        while (GeneralData.isReady() == false)
        {
        }

        movableLimbs = GeneralData.getHingeLimbs("Hand Real");
        rigidBodies = new Rigidbody[movableLimbs.Length];
        limb_velocities = new float[movableLimbs.Length];
        limbData = new float[movableLimbs.Length * 2];

        for (int i = 0; i < movableLimbs.Length; i++)
        {
            rigidBodies[i] = ((GameObject) movableLimbs[i]).GetComponent(typeof(Rigidbody)) as Rigidbody;
        }

        print("Started the threads. Detected " + movableLimbs.Length.ToString() + " controllable limbs.");

        thread = new Thread(this.ConnectionThread);
        thread.Start();
    }

    private void ConnectionThread()
    {
        // Model related logic
        connection = new ClientConnectionHandler();
        process = new Process();

        // Calls python training script.
        process.StartInfo.FileName = @"C:\Users\Michael\AppData\Local\Microsoft\WindowsApps\python.exe";
        string scriptPath = @"C:\Git\Virtual-Hand\PythonScripts\HandController.py";
        // string modelName = "Real"; // TODO, remove hard coded at some point
        // string modelName = "RealData15_shifted20"; // TODO, remove hard coded at some point
        string modelName = "T180_D6"; // TODO, remove hard coded at some point
        process.StartInfo.Arguments = scriptPath;

        // Starts the process
        print("Starting the process: " + process.StartInfo.FileName);
        process.StartInfo.UseShellExecute = false;
        process.StartInfo.CreateNoWindow = true;
        process.Start();
        System.Threading.Thread.Sleep(6000);

        connection.println(modelName);

        print("Started the Python process. ");
        print("From_Python: '" + connection.readline() + "'");
        print("From_Python: '" + connection.readline() + "'");
        for (int i = 0; i < 15; i++) // TODO, remove this hard coding at some point
        {
            print("From Python: '" + connection.readline() + "'");
            connection.println("Keep alive message.");
        }

        print("From_Python: '" + connection.readline() + "'");
        print("From_Python: '" + connection.readline() + "'");

        running = true;
        while (running)
        {
            // Sends the command to proceed with th next frame
            connection.println("next");
            // Sends current limb data
            connection.println(GetStringLimbData_forThread());
            // Receives the velocities to apply
            limb_velocities = GeneralData.string2floatArray(connection.readline());
        }
    }

    /*
 * GetStringTorques
 * Organizes the retrieval of the unity hand angles between the main and logic thread.
 */
    private string GetStringLimbData_forThread()
    {
        string stringLimbData = "";
        lock (limb_velocities_locker)
        {
            for (int i = 0; i < limbData.Length; i++)
            {
                float angle = limbData[i];

                if (i % 2 == 0) // Only converts angular position from degrees to radians
                {
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

                    angle *= 0.01745329252f;
                }

                stringLimbData += angle.ToString() + " ";
            }
        }

        stringLimbData = stringLimbData.TrimEnd(' ');
        return stringLimbData;
    }

    // Update is called once per frame
    void Update()
    {
        // Continuously records and applies existing velocities to the limbs
        lock (limb_velocities_locker)
        {
            for (int i = 0; i < movableLimbs.Length; i++)
            {
                // Records the limb data to send to the Python Script
                limbData[i * 2] = movableLimbs[i].transform.localEulerAngles.x;
                limbData[i * 2 + 1] = rigidBodies[i].angularVelocity.x;

                if (movableLimbs[i].transform.localEulerAngles.x < 0)
                {
                    movableLimbs[i].transform.localEulerAngles = new Vector3(0, 0, 0);
                    rigidBodies[i].angularVelocity = new Vector3(0, 0, 0);
                }
                else if (i % 3 != 0
                ) // TODO, if this works, then make a function that searches for the nearest parent with a rigid body (if exists)
                {
                    rigidBodies[i].angularVelocity =
                        new Vector3(
                            movableLimbs[i].transform.parent.GetComponent<Rigidbody>().angularVelocity.x +
                            limb_velocities[i], 0, 0);
                }
                else
                {
                    // Applies the current velocity to the hand
                    rigidBodies[i].angularVelocity = new Vector3(limb_velocities[i], 0, 0);
                }
            }
        }
    }

    private void OnDestroy()
    {
        connection.println("quit");
        running = false; // Stops the thread loop
        process.Close();
        connection.stop();
        print("Stopped.");
    }
}