; Inno Setup script for Shannon.
; Produces ShannonSetup.exe - what the user double-clicks.
; Built by .github/workflows/build-windows.yml on a windows-latest runner.

#define AppName    "Shannon"
#define AppVersion GetEnv("SHANNON_VERSION")
#define Publisher  "Shannon"

[Setup]
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#Publisher}
; C:\Shannon, NOT Program Files and NOT Documents. Documents is inside
; OneDrive on most Windows 11 machines, which would sync clinical audio to
; Microsoft. Program Files needs admin and makes the models hard to replace.
DefaultDirName=C:\Shannon
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
DisableDirPage=no
; No admin rights needed - the whole point is that she can install it herself.
PrivilegesRequired=lowest
OutputBaseFilename=ShannonSetup
Compression=lzma2/max
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
LicenseFile=..\LICENSE.txt
DisableWelcomePage=no

[Files]
Source: "..\transcribe.py";        DestDir: "{app}"; Flags: ignoreversion
Source: "..\README.md";            DestDir: "{app}"; Flags: ignoreversion
Source: "Shannon.bat";             DestDir: "{app}"; Flags: ignoreversion
Source: "..\build\python\*";       DestDir: "{app}\python"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\models\*";             DestDir: "{app}\models"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Shannon";            Filename: "{app}\Shannon.bat"; WorkingDir: "{app}"
Name: "{userdesktop}\Shannon";      Filename: "{app}\Shannon.bat"; WorkingDir: "{app}"; Tasks: desktopicon
Name: "{group}\Uninstall Shannon";  Filename: "{uninstallexe}"

[Tasks]
Name: "desktopicon"; Description: "Create a shortcut on the Desktop"; GroupDescription: "Shortcuts:"

[Run]
Filename: "{app}\Shannon.bat"; Description: "Transcribe a recording now"; Flags: postinstall nowait skipifsilent shellexec

[Messages]
WelcomeLabel2=This installs Shannon, which transcribes interview recordings on this computer.%n%nNothing is uploaded. No internet connection is needed once it is installed.%n%nIt will be installed to C:\Shannon.
